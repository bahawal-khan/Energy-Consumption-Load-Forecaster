"""
app.py
======
Phase 4 of the Energy Consumption Forecasting system: a polished, responsive
Streamlit UI on top of the ANN trained by train.py.

Run locally:
    streamlit run app.py

Requires artifacts produced by `python train.py` to be present in
`artifacts/` (model.h5, feature_scaler.joblib, target_scaler.joblib). If they
are missing, the app tells you exactly what to run instead of crashing.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.validation import validate_all, LIMITS

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.h5")
FEATURE_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "feature_scaler.joblib")
TARGET_SCALER_PATH = os.path.join(ARTIFACTS_DIR, "target_scaler.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="Energy Load Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme system (custom, toggle-able at runtime -- Streamlit's native theme
# is set once via config.toml, so a live Dark/Light switch is implemented
# with injected CSS variables driven by session_state)
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#0E1117", "bg_card": "#161B22", "bg_card_alt": "#1C2128",
        "text": "#E6EDF3", "text_muted": "#8B949E", "border": "#30363D",
        "accent": "#3FB1F5", "accent2": "#7C5CFF", "good": "#3FD68A",
        "warn": "#F5B93F", "danger": "#F5573F", "chart_grid": "#30363D",
    },
    "light": {
        "bg": "#F6F8FA", "bg_card": "#FFFFFF", "bg_card_alt": "#F0F3F6",
        "text": "#1F2328", "text_muted": "#57606A", "border": "#D0D7DE",
        "accent": "#0969DA", "accent2": "#8250DF", "good": "#1A7F37",
        "warn": "#9A6700", "danger": "#CF222E", "chart_grid": "#D0D7DE",
    },
}


def inject_theme_css(mode: str) -> None:
    t = THEMES[mode]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {t['bg']};
            color: {t['text']};
        }}
        [data-testid="stSidebar"] {{
            background-color: {t['bg_card']};
            border-right: 1px solid {t['border']};
        }}
        h1, h2, h3, h4, p, span, label, .stMarkdown {{
            color: {t['text']};
        }}
        .kpi-card {{
            background: linear-gradient(135deg, {t['bg_card']} 0%, {t['bg_card_alt']} 100%);
            border: 1px solid {t['border']};
            border-radius: 16px;
            padding: 28px 24px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.12);
        }}
        .kpi-label {{
            color: {t['text_muted']};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            color: {t['accent']};
            font-size: 2.6rem;
            font-weight: 800;
            line-height: 1.1;
        }}
        .kpi-sub {{
            color: {t['text_muted']};
            font-size: 0.9rem;
            margin-top: 8px;
        }}
        .info-card {{
            background: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
        }}
        .badge-good {{ background: {t['good']}22; color: {t['good']}; border: 1px solid {t['good']}55; }}
        .badge-warn {{ background: {t['warn']}22; color: {t['warn']}; border: 1px solid {t['warn']}55; }}
        .badge-danger {{ background: {t['danger']}22; color: {t['danger']}; border: 1px solid {t['danger']}55; }}
        div[data-testid="stMetric"] {{
            background: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 12px 16px;
        }}
        @media (max-width: 640px) {{
            .kpi-card {{
                padding: 18px 16px;
                border-radius: 12px;
            }}
            .kpi-value {{
                font-size: 1.9rem;
            }}
            .kpi-label {{
                font-size: 0.75rem;
            }}
            .info-card {{
                padding: 14px 16px;
            }}
            h1 {{
                font-size: 1.5rem !important;
                line-height: 1.25 !important;
            }}
            div[data-testid="stMetric"] {{
                padding: 8px 10px;
            }}
            div[data-testid="stMetricValue"] {{
                font-size: 1.3rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    """Convert a '#RRGGBB' hex string to an 'rgba(r,g,b,a)' string that
    Plotly's fillcolor validator accepts (it rejects 8-digit hex+alpha)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    fscaler = joblib.load(FEATURE_SCALER_PATH)
    tscaler = joblib.load(TARGET_SCALER_PATH)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, fscaler, tscaler, metrics


def artifacts_available() -> bool:
    return all(os.path.exists(p) for p in [MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH])


def predict_load(model, fscaler, tscaler, temperature, humidity, wind_speed, hour, day_of_week, is_holiday) -> float:
    """Feature order MUST match src.data_pipeline.FEATURE_COLS exactly."""
    x = np.array([[temperature, humidity, wind_speed, hour, day_of_week, is_holiday]], dtype=float)
    x_scaled = fscaler.transform(x)
    y_scaled = model.predict(x_scaled, verbose=0)
    y = tscaler.inverse_transform(y_scaled)
    return float(y[0, 0])


def predict_24h_curve(model, fscaler, tscaler, temperature, humidity, wind_speed, day_of_week, is_holiday) -> pd.DataFrame:
    """Sweep hour 0-23 holding other inputs fixed, for the trend chart."""
    hours = list(range(24))
    X = np.array([[temperature, humidity, wind_speed, h, day_of_week, is_holiday] for h in hours], dtype=float)
    X_scaled = fscaler.transform(X)
    y_scaled = model.predict(X_scaled, verbose=0)
    y = tscaler.inverse_transform(y_scaled).flatten()
    return pd.DataFrame({"hour": hours, "predicted_kw": y})


# ---------------------------------------------------------------------------
# Sidebar: theme toggle + inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    theme_choice = st.radio("Appearance", ["🌙 Dark", "☀️ Light"], horizontal=True,
                             index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = "dark" if "Dark" in theme_choice else "light"

    st.markdown("---")
    st.markdown("### 🔧 Forecast Inputs")

    temp_raw = st.slider(
        "Temperature (°C)", min_value=LIMITS["temperature"]["min"], max_value=LIMITS["temperature"]["max"],
        value=24.0, step=0.5,
    )
    humidity_raw = st.slider(
        "Humidity (%)", min_value=LIMITS["humidity"]["min"], max_value=LIMITS["humidity"]["max"],
        value=55.0, step=1.0,
    )
    wind_raw = st.slider(
        "Wind Speed (m/s)", min_value=LIMITS["wind_speed"]["min"], max_value=LIMITS["wind_speed"]["max"],
        value=3.0, step=0.1,
    )
    hour_raw = st.slider("Hour of Day", min_value=0, max_value=23, value=18, step=1)
    day_raw = st.selectbox("Day of Week", options=list(range(7)), format_func=lambda i: DAY_NAMES[i], index=4)
    holiday_raw = st.selectbox("Is Holiday?", options=["No", "Yes"], index=0)

    st.markdown("---")
    manual_override = st.checkbox("✍️ Enter values manually instead (type-validation demo)")
    manual_payload = {}
    if manual_override:
        manual_payload = {
            "temperature": st.text_input("Temperature (°C)", value="24"),
            "humidity": st.text_input("Humidity (%)", value="55"),
            "wind_speed": st.text_input("Wind Speed (m/s)", value="3"),
            "hour": st.text_input("Hour (0-23)", value="18"),
            "day_of_week": st.text_input("Day of Week (0=Mon..6=Sun)", value="4"),
            "is_holiday": st.text_input("Holiday? (Yes/No)", value="No"),
        }

inject_theme_css(st.session_state.theme)
t = THEMES[st.session_state.theme]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"<h1 style='margin-bottom:0;'>⚡ Energy Consumption & Load Forecaster</h1>"
    f"<p style='color:{t['text_muted']}; font-size:1.05rem; margin-top:4px;'>"
    f"Pure ANN (Multi-Layer Perceptron) · Trained on the Fedesoriano Electric Power Consumption dataset</p>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

if not artifacts_available():
    st.warning(
        "**No trained model found yet.** Run the training pipeline first:\n\n"
        "```bash\npython train.py --data data/powerconsumption.csv\n```\n\n"
        "This will clean the data, engineer features, train the ANN with early stopping, "
        "and save `artifacts/model.h5` + scalers. If you don't have the real Kaggle CSV yet, "
        "`train.py` will auto-generate a synthetic stand-in dataset so you can try the full app immediately "
        "— just swap in the real file later for production accuracy.",
        icon="⚠️",
    )
    st.stop()

model, fscaler, tscaler, saved_metrics = load_artifacts()

# ---------------------------------------------------------------------------
# Resolve + validate inputs (Phase 3)
# ---------------------------------------------------------------------------
if manual_override:
    payload = {
        "temperature": manual_payload["temperature"],
        "humidity": manual_payload["humidity"],
        "wind_speed": manual_payload["wind_speed"],
        "hour": manual_payload["hour"],
        "day_of_week": manual_payload["day_of_week"],
        "is_holiday": manual_payload["is_holiday"],
    }
else:
    payload = {
        "temperature": temp_raw,
        "humidity": humidity_raw,
        "wind_speed": wind_raw,
        "hour": hour_raw,
        "day_of_week": day_raw,
        "is_holiday": holiday_raw,
    }

ok, clean, errors = validate_all(payload)

if not ok:
    st.error("**Invalid input detected** — please fix the following before a forecast can be generated:")
    for e in errors:
        st.markdown(f"- {e}")
    st.stop()

temperature = clean["temperature"]
humidity = clean["humidity"]
wind_speed = clean["wind_speed"]
hour = clean["hour"]
day_of_week = clean["day_of_week"]
is_holiday = clean["is_holiday"]

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
try:
    predicted_kw = predict_load(model, fscaler, tscaler, temperature, humidity, wind_speed, hour, day_of_week, is_holiday)
    curve_df = predict_24h_curve(model, fscaler, tscaler, temperature, humidity, wind_speed, day_of_week, is_holiday)
except Exception as e:
    st.error(f"⚠️ The model failed to produce a prediction from these inputs: `{e}`. "
              "This has been safely caught — no crash occurred. Please adjust your inputs and try again.")
    st.stop()

peak_kw = curve_df["predicted_kw"].max()
baseline_kw = curve_df["predicted_kw"].min()
is_peak_hour = predicted_kw >= (baseline_kw + 0.75 * (peak_kw - baseline_kw))

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1.4, 1, 1])

with col1:
    badge_class = "badge-danger" if is_peak_hour else "badge-good"
    badge_text = "⚠ Peak-Hour Strain" if is_peak_hour else "✓ Stable Baseline Usage"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Predicted Load — {DAY_NAMES[day_of_week]}, {hour:02d}:00</div>
            <div class="kpi-value">{predicted_kw:,.0f} kW</div>
            <div class="kpi-sub">
                <span class="badge {badge_class}">{badge_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric("24h Peak Forecast", f"{peak_kw:,.0f} kW")
    st.metric("24h Baseline Forecast", f"{baseline_kw:,.0f} kW")

with col3:
    if saved_metrics:
        st.metric("Model Test MAE", f"{saved_metrics.get('mae_kw', 0):,.0f} kW")
        st.metric("Model Test R²", f"{saved_metrics.get('r2', 0):.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 24-hour trend chart
# ---------------------------------------------------------------------------
st.markdown("### 📈 Predicted Load Across a 24-Hour Cycle")
st.caption("Holding temperature, humidity, wind speed, and day fixed at your selected values — sweeping hour 0–23.")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=curve_df["hour"], y=curve_df["predicted_kw"],
    mode="lines+markers",
    line=dict(color=t["accent"], width=3, shape="spline"),
    marker=dict(size=6, color=t["accent"]),
    fill="tozeroy",
    fillcolor=hex_to_rgba(t["accent"], 0.13),
    name="Predicted Load",
))
fig.add_trace(go.Scatter(
    x=[hour], y=[predicted_kw],
    mode="markers",
    marker=dict(size=14, color=t["danger"], symbol="star"),
    name="Your Selection",
))
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=t["text"]),
    xaxis=dict(title="Hour of Day", gridcolor=t["chart_grid"], dtick=2),
    yaxis=dict(title="Predicted Power (kW)", gridcolor=t["chart_grid"]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=30, b=10, l=10, r=10),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Context cards
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
        <div class="info-card">
            <b>Input Snapshot</b><br><br>
            🌡️ Temperature: {temperature}°C<br>
            💧 Humidity: {humidity}%<br>
            🌬️ Wind Speed: {wind_speed} m/s<br>
            🕐 Hour: {hour:02d}:00<br>
            📅 Day: {DAY_NAMES[day_of_week]}<br>
            🏖️ Holiday: {"Yes" if is_holiday else "No"}
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    strain_pct = (predicted_kw - baseline_kw) / max(peak_kw - baseline_kw, 1e-6) * 100
    st.markdown(
        f"""
        <div class="info-card">
            <b>Grid Strain Context</b><br><br>
            This forecast sits at <b>{strain_pct:.0f}%</b> of today's predicted peak-to-baseline range.<br><br>
            {"⚠️ Consider load-shifting or demand-response measures during this window."
              if is_peak_hour else "✓ Grid conditions are comfortably within normal operating range."}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("ℹ️ About this model"):
    st.write(
        "This is a **pure feed-forward Artificial Neural Network (MLP)** built with TensorFlow/Keras — "
        "no CNN or RNN/LSTM components. It was trained on chronologically-split data (no shuffling) to "
        "avoid time-series leakage, with Dropout, Early Stopping, and best-checkpoint saving to control "
        "overfitting. Inputs are validated against physical/meteorological limits before every prediction."
    )
    if saved_metrics:
        st.json(saved_metrics)