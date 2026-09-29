import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgriYield Insights", page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #17342a 0%, #0c1f1b 32%, #08151b 100%);
        color: #f3f7f9;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.05em;
        margin-bottom: 0.15rem !important;
        color: #f8fbf9 !important;
    }
    .subtitle {
        font-size: 1.16rem;
        color: #dfeee6;
        margin-bottom: 1.5rem;
    }
    .premium-card {
        background: rgba(17, 33, 27, 0.82);
        border: 1px solid rgba(146, 217, 176, 0.18);
        border-radius: 18px;
        padding: 1.2rem 1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        backdrop-filter: blur(6px);
    }
    .side-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.3rem;
        font-weight: 800;
        color: #e8fff1;
        margin-bottom: 0.5rem;
        padding: 0.5rem 0.6rem;
        background: rgba(24, 52, 41, 0.95);
        border: 1px solid rgba(136, 204, 160, 0.2);
        border-radius: 12px;
    }
    .quote-box {
        background: linear-gradient(135deg, rgba(37,82,60,0.85), rgba(16,33,24,0.84));
        border: 1px solid rgba(160, 228, 185, 0.2);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        color: #e8fff1;
        font-size: 1.02rem;
        font-weight: 600;
        line-height: 1.6;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 8px 25px rgba(31, 71, 55, 0.2);
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(21, 39, 31, 0.9), rgba(14, 24, 20, 0.9));
        border: 1px solid rgba(159, 229, 177, 0.18);
        border-radius: 18px;
        padding: 1rem 1.15rem;
        box-shadow: 0 16px 26px rgba(0, 0, 0, 0.18);
        min-height: 110px;
    }
    .kpi-label {
        color: #c9efd6;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        color: #f5fff8;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-note {
        color: #9fe4b0;
        font-size: 0.8rem;
        margin-top: 0.45rem;
    }
    .chart-shell {
        background: linear-gradient(180deg, rgba(18, 35, 28, 0.9), rgba(11, 18, 17, 0.9));
        border: 1px solid rgba(160, 229, 185, 0.18);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 18px 35px rgba(0,0,0,0.2);
    }
    .stSelectbox > div > div,
    .stSlider > div,
    .stTextInput > div > div {
        background: rgba(15, 34, 27, 0.8);
        border: 1px solid rgba(148, 215, 170, 0.2);
        border-radius: 12px;
        color: #f3f7f9;
    }
    .stButton > button {
        background: linear-gradient(135deg, #30c685 0%, #1e9f6f 100%);
        color: white;
        border: none;
        border-radius: 14px;
        font-size: 1.08rem;
        font-weight: 700;
        height: 3.1rem;
        box-shadow: 0 15px 28px rgba(48, 198, 133, 0.28);
    }
    .stButton > button:hover {
        filter: brightness(1.04);
        transform: translateY(-1px);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .stSuccess {
        border-radius: 16px;
        background: rgba(13, 38, 30, 0.8);
        border: 1px solid rgba(105, 209, 148, 0.3);
    }
    .stInfo {
        border-radius: 16px;
        border: 1px solid rgba(97, 152, 212, 0.35);
        background: rgba(11, 28, 36, 0.75);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>🌾 AgriYield Insights</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Smart crop yield prediction for Indian states, districts, crops, and seasons.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='quote-box'>🌱  With You From Seed to Yield</div>",
    unsafe_allow_html=True,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "crop_yield_model.pkl"))
    district_freq = joblib.load(os.path.join(MODEL_DIR, "district_freq_map.pkl"))
    meta = joblib.load(os.path.join(MODEL_DIR, "app_meta.pkl"))
    return model, district_freq, meta

model, district_freq, meta = load_artifacts()

# User greeting and branding
user_name = st.text_input("Enter your name:", placeholder="e.g. Rahul")
farmer_name = user_name.strip() if user_name and user_name.strip() else "Farmer"

if user_name:
    st.write(f"### 👋 Welcome, {farmer_name}! Select your field details below:")
else:
    st.write("### 👋 Welcome! Select your field details below:")

st.markdown(
    "Predict crop yield (tonnes/hectare) for a given **state, district, crop, season, and year** using agricultural patterns from India."
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.markdown('<div class="kpi-card"><div class="kpi-label">Crop Types</div><div class="kpi-value">'+str(len(meta['crops']))+'</div><div class="kpi-note">Variety tracked</div></div>', unsafe_allow_html=True)
with metric_col2:
    st.markdown('<div class="kpi-card"><div class="kpi-label">Districts</div><div class="kpi-value">'+str(len(meta['districts_by_state']))+'</div><div class="kpi-note">Regional patterns</div></div>', unsafe_allow_html=True)
with metric_col3:
    st.markdown('<div class="kpi-card"><div class="kpi-label">Model</div><div class="kpi-value">'+str(meta['best_model_name'])+'</div><div class="kpi-note">Best performer</div></div>', unsafe_allow_html=True)

trend_data = pd.DataFrame(
    {
        "Season": ["Kharif", "Rabi", "Zaid", "Annual"],
        "Yield": [2.2, 2.8, 3.5, 4.1],
        "Target": [2.0, 2.6, 3.2, 3.8],
    }
)

st.subheader("Yield Overview")
st.markdown("A quick performance view of the major seasonal yield pattern.")


with st.sidebar:
    st.markdown('<div class="side-logo">🌾 AgriYield</div>', unsafe_allow_html=True)
    st.markdown("**Precision agriculture dashboard**")
    st.markdown("---")
    st.header("About")
    st.write(
        f"**Model:** {meta['best_model_name']}\n\n"
        f"**Test R²:** {meta['metrics']['R2']:.3f}\n\n"
        f"**Test RMSE:** {meta['metrics']['RMSE']:.2f} t/ha\n\n"
        "Trained on state/district-level crop production records across India, 1997–2020."
    )
    st.caption(
        "Note: this model uses only location, crop, season, and year as inputs — "
        "it does not have access to weather/rainfall data, so predictions represent "
        "a historical-pattern estimate, not a weather-adjusted forecast."
    )
    st.markdown("---")
    st.markdown(
        "<div class='quote-box'>“The soil is the foundation of every thriving nation.”</div>",
        unsafe_allow_html=True,
    )

st.subheader("Enter details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", meta["states"], index=meta["states"].index("Karnataka") if "Karnataka" in meta["states"] else 0)
    districts = meta["districts_by_state"].get(state, [])
    district = st.selectbox("District", districts)
    crop = st.selectbox("Crop", meta["crops"], index=meta["crops"].index("Rice") if "Rice" in meta["crops"] else 0)

with col2:
    season = st.selectbox("Season", meta["seasons"])
    year = st.slider("Year", min_value=meta["year_range"][0], max_value=meta["year_range"][1], value=2026)

st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
st.info(
    f"{farmer_name}'s farm profile: {crop} cultivation in {district}, {state}, during the {season.lower()} season in {year}. "
    "This context helps interpret the yield estimate for the selected agricultural conditions."
)
st.markdown("</div>", unsafe_allow_html=True)

def safe_district_frequency(district_name, district_lookup):
    if isinstance(district_lookup, dict):
        values = list(district_lookup.values())
        if values:
            return float(sum(values) / len(values))
    return 0.0


if st.button("Predict Yield", type="primary", use_container_width=True):
    freq = district_freq.get(district)
    if freq is None:
        freq = safe_district_frequency(district, district_freq)

    input_df = pd.DataFrame([{
        "state": state,
        "crop": crop,
        "season": season,
        "year": year,
        "district_freq": freq
    }])

    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 0)  # yield can't be negative

    st.success(
        f"### Predicted Yield for {farmer_name}: **{prediction:.2f} tonnes/hectare**"
    )
    st.caption(
        f"{crop} · {season} season · {district}, {state} · {year}"
    )

    benchmark = max(prediction * 0.9, 0.5)
    optimistic = max(prediction * 1.1, 0.5)
    chart_values = [prediction, benchmark, optimistic]
    chart_labels = ["Current\nEstimate", "Expected\nBenchmark", "Optimistic\nTarget"]

    fig = plt.figure(figsize=(10, 6), facecolor="#0b1a17")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b1a17")

    x_pos = np.arange(len(chart_values))
    y_pos = np.zeros(len(chart_values))
    z_pos = np.zeros(len(chart_values))
    dx = np.full(len(chart_values), 0.6)
    dy = np.full(len(chart_values), 0.6)
    dz = np.array(chart_values)
    colors = ["#7ae7a6", "#5cc8ff", "#f7d36c"]

    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, edgecolor="white", linewidth=0.7, alpha=0.92)
    ax.set_xticks(x_pos + 0.3)
    ax.set_xticklabels(chart_labels, rotation=10, ha="right", color="#dfeee6")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_zlabel("Yield (t/ha)", color="#dfeee6", labelpad=12)
    ax.tick_params(axis="z", colors="#dfeee6")
    ax.tick_params(axis="x", colors="#dfeee6")
    ax.grid(True, alpha=0.2)
    ax.set_title("Premium 3D Yield View", color="#f8fbf9", pad=18, fontsize=14)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.08, 0.14, 0.12, 0.96))
        pane.set_edgecolor((0.28, 0.52, 0.42, 0.9))

    st.markdown("<div class='chart-shell'>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Easy reading: the green bar is your current predicted yield, the blue bar is the expected benchmark, "
        "and the yellow bar is an optimistic target. Higher bars mean a better expected output for the selected crop and season."
    )

    st.markdown("---")
    st.markdown(
        f"**Farm insight for {farmer_name}:** Based on the selected conditions in **{district}, {state}**, "
        f"the expected output for **{crop}** in the **{season.lower()}** season is around **{prediction:.2f} t/ha**. "
        "This estimate reflects historical patterns and should be interpreted as a planning indicator rather than a weather-adjusted forecast."
    )

st.markdown("---")
st.caption(
    "Built as a data science portfolio project · Data: India crop production statistics (1997–2020) · "
    "Models compared: Linear Regression, Random Forest, XGBoost"
)