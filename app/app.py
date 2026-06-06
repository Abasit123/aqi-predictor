import os
import sys
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from utils.data_loader  import (
    load_latest_row,
    load_last_3_days,
    load_model_metrics
)
from utils.model_loader import (
    load_models_from_registry,
    get_best_model_names_from_mongo
)
from src.features.feature_engineering import get_features

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="AQI Forecast · Hyderabad",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: #3b82f6;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6b7a99 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #e8eaf0 !important;
}

/* ── Section headings ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d45;
}

/* ── Forecast card ── */
.forecast-card {
    background: linear-gradient(135deg, #111827 0%, #151e2e 100%);
    border-radius: 16px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    border: 1px solid #1e2d45;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.forecast-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.forecast-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.forecast-card .horizon-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7a99;
    margin-bottom: 0.8rem;
}
.forecast-card .aqi-value {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.4rem 0;
}
.forecast-card .category-label {
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.4rem 0;
}
.forecast-card .change-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.6rem;
    color: #6b7a99;
}
.forecast-card .model-badge {
    display: inline-block;
    margin-top: 1rem;
    background: #0f172a;
    border: 1px solid #1e2d45;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    color: #3b82f6;
}

/* ── Alert box ── */
.alert-box {
    background: linear-gradient(135deg, #1a0a0a, #2a1010);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
}

/* ── Metrics table ── */
.metrics-row {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
}

/* ── Divider ── */
hr {
    border-color: #1e2d45 !important;
    margin: 1.5rem 0 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #3b82f6 !important;
}
</style>
""", unsafe_allow_html=True)


# ── AQI helpers ───────────────────────────────────────────────

AQI_LEVELS = [
    (50,  "Good",                 "#22c55e", "#052e16"),
    (100, "Moderate",             "#eab308", "#1c1500"),
    (150, "Unhealthy for Groups", "#f97316", "#1c0a00"),
    (200, "Unhealthy",            "#ef4444", "#1c0000"),
    (300, "Very Unhealthy",       "#a855f7", "#120011"),
    (999, "Hazardous",            "#dc2626", "#110000"),
]

def aqi_meta(val: float) -> dict:
    for threshold, cat, color, bg in AQI_LEVELS:
        if val <= threshold:
            return {"category": cat, "color": color, "bg": bg}
    return {"category": "Hazardous", "color": "#dc2626", "bg": "#110000"}

AQI_ADVICE = {
    "Good":                 "Air quality is satisfactory. Great day for outdoor activities.",
    "Moderate":             "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.",
    "Unhealthy for Groups": "Sensitive groups should reduce outdoor activity. Others are unaffected.",
    "Unhealthy":            "Everyone should reduce prolonged outdoor exertion.",
    "Very Unhealthy":       "Everyone should avoid outdoor activity. Stay indoors.",
    "Hazardous":            "Health emergency. Everyone should remain indoors.",
}


# ── Cached loaders ────────────────────────────────────────────

@st.cache_resource(ttl=3600)
def load_models():
    best_names = get_best_model_names_from_mongo()
    from utils.model_loader import BEST_MODELS
    BEST_MODELS.update(best_names)
    return load_models_from_registry()


@st.cache_data(ttl=300)
def load_data():
    # Load base structure and history from local databases
    latest  = load_latest_row()
    history = load_last_3_days()
    metrics = load_model_metrics()
    
    # Live data extraction straight from Open-Meteo APIs for Hyderabad, Sindh
    try:
        lat, lon = 25.3960, 68.3578
        
        # 1. Fetch live meteorological features
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        weather_data = requests.get(weather_url, timeout=10).json()
        
        # 2. Fetch live environmental AQI indexes
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=european_aqi,pm2_5,pm10&forecast_days=1&timezone=Asia/Karachi"
        aqi_data = requests.get(aqi_url, timeout=10).json()

        
        # Safely map external payload data straight into the active dictionary schema
        if "current" in weather_data:
            latest["temperature_c"]     = float(weather_data["current"]["temperature_2m"])
            latest["humidity_pct"]      = float(weather_data["current"]["relative_humidity_2m"])
            latest["wind_speed_kmh"]    = float(weather_data["current"]["wind_speed_10m"])
            
        
        if "hourly" in aqi_data:
            latest["aqi"]  = float(aqi_data["hourly"]["european_aqi"][0])
            latest["pm25"] = float(aqi_data["hourly"]["pm2_5"][0])
            latest["pm10"] = float(aqi_data["hourly"]["pm10"][0])
            
        print("✓ Real-time conditions synchronized with Open-Meteo APIs.")
        
    except Exception as api_err:
        # Fallback tracking indicator if Open-Meteo endpoints drop offline
        print(f"⚠️ Live Open-Meteo sync failed ({api_err}). Reverting to database cache state.")

    return latest, history, metrics


# ── Predictions ───────────────────────────────────────────────

def predict(models: dict, latest_row: dict) -> dict:
    predictions = {}
    for horizon, model in models.items():
        if model is None:
            predictions[horizon] = None
            continue
        try:
            features = get_features(horizon)
            X        = pd.DataFrame([latest_row])[features]
            pred     = model.predict(X)[0]
            predictions[horizon] = round(float(pred), 1)
        except Exception as e:
            predictions[horizon] = None
    return predictions


# ── Charts ────────────────────────────────────────────────────

def make_gauge(val: float, title: str, size: int = 200) -> go.Figure:
    meta = aqi_meta(val)
    fig  = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = val,
        number= {"font": {"size": 36, "color": meta["color"],
                          "family": "Syne"}},
        title = {"text": title,
                 "font": {"size": 13, "color": "#6b7a99",
                          "family": "DM Mono"}},
        gauge = {
            "axis": {
                "range": [0, 300],
                "tickwidth": 1,
                "tickcolor": "#1e2d45",
                "tickfont": {"size": 9, "color": "#6b7a99"},
            },
            "bar":   {"color": meta["color"], "thickness": 0.25},
            "bgcolor": "#0f172a",
            "bordercolor": "#1e2d45",
            "borderwidth": 1,
            "steps": [
                {"range": [0,   50],  "color": "#052e16"},
                {"range": [50,  100], "color": "#1c1500"},
                {"range": [100, 150], "color": "#1c0a00"},
                {"range": [150, 200], "color": "#1c0000"},
                {"range": [200, 300], "color": "#120011"},
            ],
        }
    ))
    fig.update_layout(
        height=size,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        font={"color": "#e8eaf0"},
    )
    return fig


def make_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # AQI area
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["aqi"],
        mode="lines+markers",
        name="AQI",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=5, color="#3b82f6"),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>AQI: %{y:.0f}<extra></extra>"
    ))

    # PM2.5 line
    if "pm25" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["pm25"],
            mode="lines",
            name="PM2.5",
            line=dict(color="#f97316", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>PM2.5: %{y:.1f}<extra></extra>"
        ))

    # Reference lines
    for level, label, color in [
        (50,  "Good",     "#22c55e"),
        (100, "Moderate", "#eab308"),
        (150, "Unhealthy","#ef4444"),
    ]:
        fig.add_hline(
            y=level, line_dash="dash",
            line_color=color, line_width=1,
            opacity=0.4,
            annotation_text=label,
            annotation_position="right",
            annotation_font={"size": 9, "color": color}
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#090d18",
        height=280,
        margin=dict(l=10, r=80, t=10, b=10),
        legend=dict(
            orientation="h", x=0, y=1.1,
            font={"size": 11, "color": "#9aa3b8"},
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#1e2d45",
            tickfont={"size": 10, "color": "#6b7a99",
                      "family": "DM Mono"},
            showline=False
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1e2d45",
            tickfont={"size": 10, "color": "#6b7a99",
                      "family": "DM Mono"},
            title=dict(text="AQI", font={"size": 11,
                               "color": "#6b7a99"}),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#3b82f6",
            font={"size": 12, "color": "#e8eaf0"}
        )
    )
    return fig


# ── Forecast card ─────────────────────────────────────────────

def forecast_card(
    horizon: str,
    pred: float,
    current: float,
    model_name: str,
    metrics_row: dict
):
    if pred is None:
        st.error(f"Model unavailable for {horizon}")
        return

    meta     = aqi_meta(pred)
    change   = pred - current
    chg_sign = "↑" if change > 0 else "↓"
    chg_col  = "#ef4444" if change > 8 else "#22c55e" if change < -8 else "#6b7a99"
    labels   = {"24h": "AVG NEXT 24 HOURS", "48h": "AVG NEXT 48 HOURS", "72h": "AVG NEXT 72 HOURS"}

    # Format model name
    model_display = model_name.replace("_", " ").title() if model_name else "—"

    # Metrics string
    mae = metrics_row.get("mae", "—")
    r2  = metrics_row.get("r2",  "—")
    metrics_str = f"MAE {mae} · R² {r2}" if mae != "—" else ""

    st.markdown(f"""
    <div class="forecast-card" style="border-color: {meta['color']}22;">
        <div style="
            position: absolute; top:0; left:0; right:0;
            height:3px; background:{meta['color']};
            border-radius:16px 16px 0 0;
        "></div>
        <div class="horizon-label">{labels.get(horizon, horizon)}</div>
        <div class="aqi-value" style="color:{meta['color']}">
            {pred:.0f}
        </div>
        <div class="category-label" style="color:{meta['color']}">
            {meta['category']}
        </div>
        <div class="change-label" style="color:{chg_col}">
            {chg_sign} {abs(change):.1f} from current
        </div>
        <div style="margin-top:0.8rem; font-size:0.78rem; color:#9aa3b8; line-height:1.4">
            {AQI_ADVICE.get(meta['category'], '')}
        </div>
        <div class="model-badge">{model_display}</div>
        {f'<div style="font-family:DM Mono,monospace;font-size:0.62rem;color:#3b5a8a;margin-top:0.4rem">{metrics_str}</div>' if metrics_str else ''}
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────

def main():

    # ── Header ───────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("""
        <div style="margin-bottom:0.2rem">
            <span style="
                font-family:'DM Mono',monospace;
                font-size:0.7rem;
                letter-spacing:0.18em;
                text-transform:uppercase;
                color:#3b82f6;
            ">Air Quality Monitoring System</span>
        </div>
        <h1 style="
            font-family:'Syne',sans-serif;
            font-weight:800;
            font-size:2.2rem;
            color:#e8eaf0;
            margin:0;
            line-height:1.1;
        ">Hyderabad, Sindh<br>
        <span style="color:#3b82f6;">AQI Forecast</span>
        </h1>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown(f"""
        <div style="text-align:right;margin-top:1rem">
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#6b7a99;letter-spacing:0.1em">LAST UPDATED</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;
                        color:#e8eaf0;font-weight:600">
                {datetime.now().strftime('%b %d, %Y')}
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.85rem;
                        color:#3b82f6">
                {datetime.now().strftime('%H:%M')} PKT
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────
    with st.spinner(""):
        try:
            models                   = load_models()
            latest, history, metrics = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return

    predictions  = predict(models, latest)
    current_aqi  = float(latest.get("aqi", 0))
    current_meta = aqi_meta(current_aqi)

    # Build metrics lookup {horizon: row_dict}
    metrics_by_horizon = {}
    if not metrics.empty and "horizon" in metrics.columns:
        for _, row in metrics.iterrows():
            metrics_by_horizon[row["horizon"]] = row.to_dict()

    # Build best model name lookup
    from utils.model_loader import BEST_MODELS
    best_names = get_best_model_names_from_mongo()
    BEST_MODELS.update(best_names)

    # ── Row 1: Current conditions ─────────────────────────────
    st.markdown('<div class="section-label">Current Conditions</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("AQI", f"{current_aqi:.0f}")
        st.markdown(
            f'<span style="font-size:0.8rem;color:{current_meta["color"]};'
            f'font-weight:600">{current_meta["category"]}</span>',
            unsafe_allow_html=True
        )
    with col2:
        st.metric("PM2.5", f"{latest.get('pm25', 0):.1f}",
                  help="μg/m³ — WHO limit: 15/day")
    with col3:
        st.metric("PM10",  f"{latest.get('pm10', 0):.1f}",
                  help="μg/m³ — WHO limit: 45/day")
    with col4:
        st.metric("Wind",  f"{latest.get('wind_speed_kmh', 0):.1f} km/h")
    with col5:
        st.metric("Humidity", f"{latest.get('humidity_pct', 0):.0f}%")
    with col6:
        st.metric("Temperature",
                  f"{latest.get('temperature_c', 0):.1f}°C")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 2: Gauge + Trend ──────────────────────────────────
    col_g, col_t = st.columns([1, 2])

    with col_g:
        st.markdown('<div class="section-label">Air Quality Index</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_gauge(current_aqi, "Current AQI", size=220),
            use_container_width=True
        )
        st.markdown(f"""
        <div style="
            background:{current_meta['bg']};
            border:1px solid {current_meta['color']}44;
            border-left:3px solid {current_meta['color']};
            border-radius:8px;
            padding:0.8rem 1rem;
            font-size:0.82rem;
            color:#c8cad4;
            line-height:1.5;
        ">
            {AQI_ADVICE.get(current_meta['category'], '')}
        </div>
        """, unsafe_allow_html=True)

    with col_t:
        st.markdown('<div class="section-label">3-Day AQI Trend · 3h Intervals</div>',
                    unsafe_allow_html=True)
        if not history.empty:
            st.plotly_chart(
                make_trend_chart(history),
                use_container_width=True
            )
            # Stats row below chart
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("3-Day Avg",  f"{history['aqi'].mean():.0f}")
            sc2.metric("3-Day Max",  f"{history['aqi'].max():.0f}")
            sc3.metric("3-Day Min",  f"{history['aqi'].min():.0f}")
            sc4.metric("Data Points",f"{len(history)}")
        else:
            st.info("No historical data available yet.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 3: Forecast cards ─────────────────────────────────
    st.markdown('<div class="section-label">AQI Forecast · ML Predictions</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, horizon in zip([c1, c2, c3], ["24h", "48h", "72h"]):
        with col:
            model_name  = BEST_MODELS.get(horizon, "linear_regression")
            metrics_row = metrics_by_horizon.get(horizon, {})
            forecast_card(
                horizon,
                predictions.get(horizon),
                current_aqi,
                model_name,
                metrics_row
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Row 4: Forecast gauges ────────────────────────────────
    st.markdown('<div class="section-label">Forecast Gauges</div>',
                unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    for col, horizon, label in zip(
        [g1, g2, g3],
        ["24h", "48h", "72h"],
        ["Next 24h", "Next 48h", "Next 72h"]
    ):
        with col:
            pred = predictions.get(horizon)
            if pred:
                st.plotly_chart(
                    make_gauge(pred, label, size=180),
                    use_container_width=True
                )

    # ── Row 5: Alerts ─────────────────────────────────────────
    alerts = [
        (h, p) for h, p in predictions.items()
        if p and p > 150
    ]
    if alerts:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">⚠ Air Quality Alerts</div>',
                    unsafe_allow_html=True)
        for horizon, pred in alerts:
            meta = aqi_meta(pred)
            st.markdown(f"""
            <div class="alert-box">
                <strong style="color:#ef4444">
                    {horizon.upper()} FORECAST ALERT
                </strong> —
                AQI {pred:.0f}
                <span style="color:{meta['color']}">
                    ({meta['category']})
                </span>
                <br>
                <span style="color:#9aa3b8;font-size:0.85rem">
                    {AQI_ADVICE.get(meta['category'], '')}
                </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Row 6: Model performance ──────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("📊 Model Performance — Best Models", expanded=False):
        if not metrics.empty:
            st.markdown('<div class="section-label" style="margin-top:0.5rem">Best Model Per Forecast Horizon</div>',
                        unsafe_allow_html=True)

            for _, row in metrics.sort_values("horizon").iterrows():
                horizon    = row.get("horizon", "—")
                model_name = row.get("model_name", "—").replace("_", " ").title()
                mae        = row.get("mae",  "—")
                rmse       = row.get("rmse", "—")
                r2         = row.get("r2",   "—")
                trained_at = row.get("trained_at", "—")
                meta       = aqi_meta({"24h": 40, "48h": 60, "72h": 80}.get(horizon, 60))

                st.markdown(f"""
                <div class="metrics-row">
                    <div>
                        <span style="
                            font-family:'DM Mono',monospace;
                            font-size:0.7rem;
                            letter-spacing:0.12em;
                            color:#3b82f6;
                            text-transform:uppercase;
                        ">{horizon}</span>
                        <span style="
                            font-size:0.95rem;
                            font-weight:600;
                            color:#e8eaf0;
                            margin-left:1rem;
                        ">{model_name}</span>
                    </div>
                    <div style="display:flex;gap:2rem;align-items:center">
                        <div style="text-align:center">
                            <div style="font-family:'DM Mono',monospace;
                                        font-size:0.62rem;color:#6b7a99;
                                        text-transform:uppercase">MAE</div>
                            <div style="font-size:1rem;font-weight:700;
                                        color:#e8eaf0">{mae}</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-family:'DM Mono',monospace;
                                        font-size:0.62rem;color:#6b7a99;
                                        text-transform:uppercase">RMSE</div>
                            <div style="font-size:1rem;font-weight:700;
                                        color:#e8eaf0">{rmse}</div>
                        </div>
                        <div style="text-align:center">
                            <div style="font-family:'DM Mono',monospace;
                                        font-size:0.62rem;color:#6b7a99;
                                        text-transform:uppercase">R²</div>
                            <div style="font-size:1rem;font-weight:700;
                                        color:#22c55e">{r2}</div>
                        </div>
                        <div style="text-align:right">
                            <div style="font-family:'DM Mono',monospace;
                                        font-size:0.62rem;color:#6b7a99;
                                        text-transform:uppercase">Trained</div>
                            <div style="font-size:0.8rem;color:#9aa3b8">
                                {trained_at}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No metrics available yet. Run the training pipeline first.")

    # ── Row 7: Raw data ───────────────────────────────────────
    with st.expander("🔍 Latest Raw Features", expanded=False):
        drop_cols = [c for c in latest.keys()
                     if "target" in str(c) or "lag" in str(c)
                     or "roll" in str(c)]
        display   = {k: v for k, v in latest.items()
                     if k not in drop_cols}
        st.dataframe(
            pd.DataFrame([display]),
            use_container_width=True
        )

    # ── Footer ────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        font-family:'DM Mono',monospace;
        font-size:0.65rem;
        letter-spacing:0.08em;
        color:#3b5a8a;
        padding:0.5rem 0;
    ">
        <span>10PEARLS AQI PREDICTOR · HYDERABAD, SINDH, PAKISTAN</span>
        <span>DATA · OPEN-METEO · MONGODB · HOPSWORKS · STREAMLIT</span>
        <span>UPDATES HOURLY VIA GITHUB ACTIONS</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()