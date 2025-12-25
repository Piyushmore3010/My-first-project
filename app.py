# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model
import tensorflow as tf
import time

st.set_page_config(layout="wide", page_title="⚡ Smart Energy Monitor", page_icon="⚡")

# -----------------------------
# Sidebar Reset & Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

# ✅ Clear Cache / Reset Button
if st.sidebar.button("♻️ Clear Cache / Reset Dashboard"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("✅ Cache cleared! Please rerun the app.")
    st.stop()

# -----------------------------
# Load Data & Models
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/simulated_3days.csv", index_col=0, parse_dates=True)
    return df

@st.cache_resource
def load_models():
    cnn = load_model("models/cnn_disagg.h5", compile=False)
    lstm = load_model("models/lstm_forecast.h5", compile=False)
    cnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    lstm.compile(optimizer='adam', loss='mse', metrics=['mae'])
    meta = joblib.load("models/meta.pkl")
    return cnn, lstm, meta

df = load_data()
cnn, lstm, meta = load_models()
APPLIANCES = meta["appliances"]

# -----------------------------
# Sidebar Selections
# -----------------------------
day = st.sidebar.selectbox("Choose Day", pd.to_datetime(df.index.date).unique())
start = st.sidebar.slider("Start Minute of Day", 0, 24*60-60, 0, 60)
window_minutes = st.sidebar.slider("Disaggregation Window (minutes)", 30, 180, 120, 10)
selected_appliances = st.sidebar.multiselect("Select Appliances", APPLIANCES, APPLIANCES)
sim_speed = st.sidebar.slider("Simulation Speed (sec/update)", 1, 10, 2)
show_realtime = st.sidebar.checkbox("Show Real-Time Simulation (optional)", value=False)

# -----------------------------
# Data Filtering
# -----------------------------
day_mask = df.index.date == pd.to_datetime(day).date()
df_day = df.loc[day_mask]
view = df_day.iloc[start:start+24*60]

# -----------------------------
# Dashboard KPIs
# -----------------------------
st.title("⚡ Smart Energy Monitoring Dashboard")
st.markdown("Monitor your household’s energy consumption and gain **appliance-level insights** in real time or full view.")

total_consumption = view["aggregate"].sum()
peak_appliance = view[APPLIANCES].sum().idxmax()
avg_usage = view[APPLIANCES].mean()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("💡 Total Consumption (Wh)", f"{total_consumption:.0f}")
kpi2.metric("📈 Peak Appliance", f"{peak_appliance}")
kpi3.metric("⚖️ Avg Appliance Usage (W)", f"{avg_usage.mean():.0f}")

st.divider()

# -----------------------------
# Aggregate Load (Normal + Simulation)
# -----------------------------
st.subheader("📊 Aggregate Load (Full View + Optional Real-Time Simulation)")

# Show normal full-day graph
fig_full = px.line(
    x=view.index, y=view["aggregate"].values,
    labels={"x": "Time", "y": "Power (W)"},
    title="Aggregate Power (W) - Full Day",
    template="plotly_dark"
)
fig_full.update_traces(line=dict(color="yellow", width=3))
st.plotly_chart(fig_full, use_container_width=True)

# Optional simulation
if show_realtime:
    st.markdown("### ⚙️ Real-Time Simulation (Aggregate Load)")
    placeholder_sim = st.empty()
    agg_values = view["aggregate"].values
    timestamps = view.index
    for i in range(1, len(agg_values)+1):
        fig_sim = px.line(
            x=timestamps[:i], y=agg_values[:i],
            labels={"x": "Time", "y": "Power (W)"},
            title="Live Aggregate Power (Simulated)",
            template="plotly_dark"
        )
        fig_sim.update_traces(line=dict(color="orange", width=3))
        placeholder_sim.plotly_chart(fig_sim, use_container_width=True)
        time.sleep(sim_speed)

st.divider()

# -----------------------------
# CNN Disaggregation
# -----------------------------
st.subheader("🔌 Appliance Disaggregation (CNN Estimate)")

def run_disagg_for_slice(view, cnn_model, window=meta["window"]):
    agg = view["aggregate"].values
    half = window // 2
    X = [agg[i-half:i+half].reshape(window, 1) for i in range(half, len(agg)-half)]
    if len(X) == 0:
        return None, None
    X = np.array(X)
    preds = cnn_model.predict(X, verbose=0)
    times = view.index[half:len(agg)-half]
    df_preds = pd.DataFrame(preds, index=times, columns=APPLIANCES)
    return df_preds, times

df_preds, times = run_disagg_for_slice(view, cnn, window=meta["window"])
if df_preds is not None:
    df_preds = df_preds[selected_appliances]
    df_preds_reset = df_preds.reset_index().rename(columns={'index': 'timestamp'})

    # Full disaggregation graph
    fig2 = px.line(
        df_preds_reset,
        x='timestamp', y=selected_appliances,
        labels={"timestamp": "Time", "value": "Power (W)"},
        title="Estimated Appliance Power (W)",
        template="plotly_dark"
    )
    fig2.update_layout(legend_title_text="Appliances")
    st.plotly_chart(fig2, use_container_width=True)

    # Optional real-time simulation
    if show_realtime:
        st.markdown("### 🔄 Real-Time Appliance Simulation")
        placeholder_realtime = st.empty()
        for i in range(1, len(df_preds_reset)+1):
            fig_sim2 = px.line(
                df_preds_reset.iloc[:i],
                x='timestamp', y=selected_appliances,
                labels={"timestamp": "Time", "value": "Power (W)"},
                title="Appliance Power (Live Simulation)",
                template="plotly_dark"
            )
            placeholder_realtime.plotly_chart(fig_sim2, use_container_width=True)
            time.sleep(sim_speed)
else:
    st.info("Pick a longer slice or smaller window for disaggregation.")

st.divider()

# -----------------------------
# LSTM Forecast
# -----------------------------
st.subheader("📈 Per-Appliance Forecast (Next Minute)")
timesteps = meta["lstm_timesteps"]
if len(df_day) >= timesteps:
    last_seq = df_day[APPLIANCES].values[-timesteps:]
    pred = lstm.predict(last_seq.reshape(1, timesteps, len(APPLIANCES)), verbose=0)[0]
    forecast_df = pd.DataFrame({"appliance": APPLIANCES, "next_min_pred_watts": pred})
    forecast_df = forecast_df[forecast_df['appliance'].isin(selected_appliances)]

    cols = st.columns(len(forecast_df))
    for col, (app, val) in zip(cols, forecast_df.values):
        col.metric(label=f"🔹 {app}", value=f"{val:.0f} W")
else:
    st.info("Not enough data for LSTM forecast.")

st.divider()

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("💡 Recommendations")
for app in selected_appliances:
    usage = avg_usage[app]
    if usage > 800:
        st.error(f"⚠️ {app}: High usage ({usage:.0f} W) — consider service or schedule changes.")
    elif usage > 300:
        st.warning(f"⚠️ {app}: Moderate usage ({usage:.0f} W) — shift to off-peak if possible.")
    else:
        st.success(f"✅ {app}: Normal usage ({usage:.0f} W)")

st.divider()

# -----------------------------
# Expandable Raw Data
# -----------------------------
with st.expander("📂 Show Raw Data"):
    st.dataframe(view)
