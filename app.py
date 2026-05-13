import streamlit as st
import plotly.graph_objects as go
import requests

# ── Import all logic from main.py ──────────────────────────────────────────
from main import get_solar_forecast, daily_summary

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Forecast",
    page_icon="☀️",
    layout="centered"
)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("☀️ Solar Power Forecast")
st.caption("7-day prediction using live weather · ML Model R² = 0.90")
st.divider()

# ── City input ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    city = st.text_input(
        "City",
        placeholder="e.g. Pathankot, Delhi, Mumbai, Jaipur...",
        label_visibility="collapsed"
    )
with col2:
    run = st.button("Get Forecast", use_container_width=True, type="primary")

# ── Run forecast ───────────────────────────────────────────────────────────
if run:
    if not city.strip():
        st.warning("Please enter a city name.")
        st.stop()

    with st.spinner(f"Fetching forecast for {city}..."):
        try:
            loc, df  = get_solar_forecast(city.strip())
            daily    = daily_summary(df)

        except ValueError as e:
            st.error(str(e))
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("No internet connection.")
            st.stop()
        except FileNotFoundError:
            st.error("solar_model.pkl not found — run Solar_Power_Model.ipynb first.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Location ───────────────────────────────────────────────────────────
    st.success(f"📍 {loc['name']}, {loc['country']}   ·   {loc['lat']:.2f}°N  {loc['lon']:.2f}°E")

    # ── Metrics ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Today",       f"{daily.iloc[0]['total_kwh']:.2f} kWh")
    m2.metric("7-Day Total", f"{daily['total_kwh'].sum():.1f} kWh")
    m3.metric("Daily Avg",   f"{daily['total_kwh'].mean():.2f} kWh")
    m4.metric("Best Day",    f"{daily['total_kwh'].max():.2f} kWh")

    st.divider()

    # ── 7-day bar chart ────────────────────────────────────────────────────
    st.subheader("7-Day Solar Generation (kWh)")
    fig_bar = go.Figure(go.Bar(
        x=[str(d) for d in daily["date"]],
        y=daily["total_kwh"],
        text=[f"{v:.1f}" for v in daily["total_kwh"]],
        textposition="outside",
        marker_color="#f5a623",
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Total: %{y:.2f} kWh<extra></extra>"
    ))
    fig_bar.update_layout(
        height=280, margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(title="kWh", gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Hourly chart ───────────────────────────────────────────────────────
    st.subheader("Hourly Breakdown")
    selected_day = st.selectbox("Select a day", daily["day_label"].tolist(),
                                label_visibility="collapsed")
    sel_date = daily.loc[daily["day_label"] == selected_day, "date"].values[0]
    df_day   = df[df["date"] == sel_date]

    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(
        x=df_day["hour"], y=df_day["predicted_kwh"],
        name="Solar (kWh)", fill="tozeroy",
        line=dict(color="#f5a623", width=2.5),
        fillcolor="rgba(245,166,35,0.12)",
        hovertemplate="Hour %{x}:00 → %{y:.3f} kWh<extra></extra>"
    ))
    fig_hour.add_trace(go.Scatter(
        x=df_day["hour"], y=df_day["ghi"] / 100,
        name="GHI ÷100 (W/m²)", line=dict(color="#cc7a00", dash="dot", width=1.5),
        hovertemplate="GHI: %{customdata:.0f} W/m²<extra></extra>",
        customdata=df_day["ghi"]
    ))
    fig_hour.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(title="Hour", dtick=2, gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title="kWh", gridcolor="rgba(0,0,0,0.08)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12), hovermode="x unified"
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    # ── Weather snapshot ───────────────────────────────────────────────────
    st.subheader(f"Weather : {selected_day}")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Avg Temp",     f"{df_day['air_temperature'].mean():.1f} °C")
    w2.metric("Avg Humidity", f"{df_day['relative_humidity'].mean():.0f} %")
    w3.metric("Avg Wind",     f"{df_day['wind_speed'].mean():.1f} m/s")
    w4.metric("Avg Cloud",    f"{df_day['cloud_opacity'].mean():.0f} %")

    st.divider()

    # ── Raw data ───────────────────────────────────────────────────────────
    with st.expander("📋 View raw forecast data"):
        show = df[[
            "timestamp", "predicted_kwh", "ghi",
            "cloud_opacity", "air_temperature", "relative_humidity", "wind_speed"
        ]].copy()
        show.columns = ["Time", "Solar (kWh)", "GHI (W/m²)",
                        "Cloud (%)", "Temp (°C)", "Humidity (%)", "Wind (m/s)"]
        show["Solar (kWh)"] = show["Solar (kWh)"].round(3)
        show["GHI (W/m²)"]  = show["GHI (W/m²)"].round(1)
        st.dataframe(show, use_container_width=True, height=260)
        st.download_button(
            "⬇️ Download CSV",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name=f"solar_forecast_{loc['name']}.csv",
            mime="text/csv"
        )