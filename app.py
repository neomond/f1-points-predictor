"""
app.py
------
Interactive Streamlit app for the F1 points-finish predictor.
Pick a driver/team/circuit and starting grid position, get the
predicted probability of finishing in the points (top 10).

Run:
    streamlit run app.py
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# app.py lives in the project root; the modules live in src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from features import build_features
from train import NUM_FEATURES, CAT_FEATURES

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "f1_raw.csv")

st.set_page_config(page_title="F1 Points Predictor", page_icon="🏎️",
                   layout="centered")

# ---- McLaren-inspired styling (papaya orange) ----
st.markdown("""
<style>
    .stApp { background-color: #15151e; color: #ffffff; }
    h1, h2, h3 { color: #ff8000; }
    .stButton>button {
        background-color: #ff8000; color: #15151e;
        font-weight: 700; border: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the saved model; if missing (e.g. fresh cloud deploy), train it."""
    if not os.path.exists(MODEL_PATH):
        # Ensure data exists too, then train + save.
        if not os.path.exists(DATA_PATH):
            from make_synthetic import main as make_data
            make_data(out=DATA_PATH)
        from evaluate import final_evaluate
        return final_evaluate()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference():
    """Latest known form per driver, to prefill history features."""
    if not os.path.exists(DATA_PATH):
        from make_synthetic import main as make_data
        make_data(out=DATA_PATH)
    df, _ = build_features(DATA_PATH)
    latest = (df.sort_values(["year", "round"])
                .groupby("driverId").tail(1)
                .set_index("driverId"))
    return df, latest


st.title("🏎️ F1 Points-Finish Predictor")
st.caption("Will this driver finish in the points (top 10)? "
           "Logistic regression trained on 2018–2025, validated with "
           "TimeSeriesSplit.")

with st.spinner("Loading model…"):
    model = load_model()
    df, latest = load_reference()

drivers = sorted(df["driverId"].unique())
circuits = sorted(df["circuitId"].unique())

col1, col2 = st.columns(2)
with col1:
    driver = st.selectbox("Driver", drivers)
    # Constructor is determined by the driver's most recent team — you
    # can't drive for two teams at once, so we lock it (no impossible combos).
    if driver in latest.index:
        constructor = latest.loc[driver, "constructorId"]
    else:
        constructor = df["constructorId"].iloc[0]
    st.text_input("Constructor", value=constructor, disabled=True,
                  help="Set automatically from the driver's current team")
with col2:
    circuit = st.selectbox("Circuit", circuits)
    grid = st.slider("Starting grid position", 1, 20, 5)

quali = st.slider("Qualifying position", 1, 20, grid)

# Pull recent-form features from the driver's latest known row
if driver in latest.index:
    row = latest.loc[driver]
    driver_avg = row["driver_avg_finish"]
    driver_recent = row["driver_recent_finish"]
    driver_rate = row["driver_points_rate"]
    cons_recent = row["constructor_recent_finish"]
    season_pts = row["season_points_so_far"]
else:
    driver_avg = driver_recent = cons_recent = 10.0
    driver_rate = 0.5
    season_pts = 0.0

X = pd.DataFrame([{
    "grid": grid, "quali_pos": quali,
    "driver_avg_finish": driver_avg,
    "driver_recent_finish": driver_recent,
    "driver_points_rate": driver_rate,
    "constructor_recent_finish": cons_recent,
    "season_points_so_far": season_pts,
    "constructorId": constructor, "circuitId": circuit,
}])[NUM_FEATURES + CAT_FEATURES]

if st.button("Predict"):
    proba = model.predict_proba(X)[0, 1]
    st.metric("Points-finish probability", f"{proba:.1%}")
    st.progress(float(proba))
    if proba >= 0.5:
        st.success(f"Likely to score points ({proba:.0%} confidence)")
    else:
        st.warning(f"Unlikely to score points ({1 - proba:.0%} confidence)")

    st.caption("Note: model trained on 2018–2025. The 2026 regulation "
               "reset shifts the competitive order, so treat 2026 "
               "predictions with extra caution (see README).")
