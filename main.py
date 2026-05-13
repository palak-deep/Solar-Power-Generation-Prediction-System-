import requests
import numpy as np
import pickle
import pandas as pd

# CONFIG
MODEL_PATH   = "D:\Solar\Model\solar_model.pkl"
GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# STEP 1 — City name → Coordinates
def get_coordinates(city: str) -> dict:
    r = requests.get(GEOCODE_URL, params={
        "name": city, "count": 1, "language": "en", "format": "json"
    }, timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"City '{city}' not found. Try a different spelling.")
    res = results[0]
    return {
        "name"    : res["name"],
        "country" : res.get("country", ""),
        "lat"     : res["latitude"],
        "lon"     : res["longitude"],
        "timezone": res.get("timezone", "auto")
    }


# STEP 2 — Fetch 7-day hourly forecast
def fetch_forecast(lat: float, lon: float, timezone: str) -> pd.DataFrame:
    r = requests.get(FORECAST_URL, params={
        "latitude"      : lat,
        "longitude"     : lon,
        "hourly"        : "temperature_2m,relativehumidity_2m,windspeed_10m,cloudcover,direct_radiation,diffuse_radiation",
        "forecast_days" : 7,
        "timezone"      : timezone,
        "windspeed_unit": "ms"
    }, timeout=10)
    r.raise_for_status()
    h = r.json()["hourly"]

    return pd.DataFrame({
        "timestamp"        : pd.to_datetime(h["time"]),
        "air_temperature"  : h["temperature_2m"],
        "relative_humidity": h["relativehumidity_2m"],
        "wind_speed"       : h["windspeed_10m"],
        "cloud_opacity"    : h["cloudcover"],
        "ghi"              : [d + f for d, f in zip(h["direct_radiation"], h["diffuse_radiation"])]
    }).dropna().reset_index(drop=True)


# STEP 3 — Feature engineering

def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build feature matrix — must match exact order used during training:
    Ghi, CloudOpacity, ghi_cloud, air_temperature, relative_humidity,
    wind_speed, hour_sin, hour_cos, month_sin, month_cos, day_of_year
    """
    ghi   = df["ghi"].to_numpy(dtype=float)
    cloud = df["cloud_opacity"].to_numpy(dtype=float)
    hour  = df["timestamp"].dt.hour.to_numpy(dtype=float)
    month = df["timestamp"].dt.month.to_numpy(dtype=float)

    return np.column_stack([
        ghi,
        cloud,
        ghi * (1 - cloud / 100),
        df["air_temperature"].to_numpy(dtype=float),
        df["relative_humidity"].to_numpy(dtype=float),
        df["wind_speed"].to_numpy(dtype=float),
        np.sin(2 * np.pi * hour  / 24),
        np.cos(2 * np.pi * hour  / 24),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        df["timestamp"].dt.dayofyear.to_numpy(dtype=float)
    ])


# STEP 4 — Load model & predict

def run_prediction(df: pd.DataFrame) -> pd.DataFrame:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    df["predicted_kwh"] = np.clip(model.predict(build_features(df)), 0, None)
    df["date"]          = df["timestamp"].dt.date
    df["hour"]          = df["timestamp"].dt.hour
    df["day_label"]     = df["timestamp"].dt.strftime("%a %d %b")
    return df

# STEP 5 — Full pipeline

def get_solar_forecast(city: str) -> tuple:
    """
    Full pipeline: city name → predictions DataFrame.
    Returns: (location dict, forecast DataFrame)
    """
    loc = get_coordinates(city)
    df  = fetch_forecast(loc["lat"], loc["lon"], loc["timezone"])
    df  = run_prediction(df)
    return loc, df



# Daily summary helper

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates hourly forecast into daily totals."""
    return df.groupby("date").agg(
        day_label = ("day_label", "first"),
        total_kwh = ("predicted_kwh", "sum"),
        peak_kwh  = ("predicted_kwh", "max"),
        avg_temp  = ("air_temperature", "mean"),
        avg_cloud = ("cloud_opacity", "mean"),
    ).reset_index()



# Standalone CLI — python main.py

if __name__ == "__main__":
    city = input("Enter city name: ").strip()
    print(f"\n🔍 Looking up: {city}")
    loc, df = get_solar_forecast(city)
    summary = daily_summary(df)
    print(f"📍 {loc['name']}, {loc['country']}  ({loc['lat']:.2f}°N, {loc['lon']:.2f}°E)\n")
    print(f"{'Date':<12} {'Day':<12} {'Total kWh':>10} {'Peak kWh':>10} {'Avg Temp':>10} {'Cloud':>8}")
    print("─" * 65)
    for _, row in summary.iterrows():
        print(f"{str(row['date']):<12} {row['day_label']:<12} "
              f"{row['total_kwh']:>10.2f} {row['peak_kwh']:>10.2f} "
              f"{row['avg_temp']:>9.1f}° {row['avg_cloud']:>7.0f}%")