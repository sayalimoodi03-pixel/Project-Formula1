import streamlit as st
import pandas as pd
import pickle
import os

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="centered"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
    color: #e0e0e0;
}

/* Title */
.f1-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #e10600, #ff4d00, #e10600);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
}

.f1-subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Card */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(225,6,0,0.2);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #e10600;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Selectbox & number input */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.05) !important;
    border-color: rgba(225,6,0,0.3) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #e10600, #b30500);
    color: white;
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    border: none;
    border-radius: 8px;
    padding: 0.8rem 2rem;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #ff1a00, #e10600);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(225,6,0,0.4);
}

/* Result boxes */
.result-podium {
    background: linear-gradient(135deg, rgba(255,215,0,0.15), rgba(255,165,0,0.05));
    border: 2px solid #ffd700;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    animation: glow 1.5s ease-in-out infinite alternate;
}
.result-no-podium {
    background: rgba(255,255,255,0.03);
    border: 2px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    margin-bottom: 0.3rem;
}
.result-sub {
    font-size: 0.9rem;
    color: #aaa;
    letter-spacing: 2px;
}

@keyframes glow {
    from { box-shadow: 0 0 10px rgba(255,215,0,0.3); }
    to   { box-shadow: 0 0 25px rgba(255,215,0,0.6); }
}

/* Divider */
hr {
    border-color: rgba(225,6,0,0.2) !important;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown('<div class="f1-title">🏎️ F1 PODIUM PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="f1-subtitle">Formula One · AI Race Analysis</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LOAD MODEL SAFELY
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = "model.pkl"
    scaler_path = "scaler.pkl"

    # Try alternate paths if needed
    alt_paths = [
        ("model.pkl",           "scaler.pkl"),
        ("Notebook/model.pkl",  "Notebook/scaler.pkl"),
    ]
    for mp, sp in alt_paths:
        if os.path.exists(mp) and os.path.exists(sp):
            with open(mp, "rb") as f:  mdl = pickle.load(f)
            with open(sp, "rb") as f:  scl = pickle.load(f)
            return mdl, scl, None

    return None, None, "❌ model.pkl or scaler.pkl not found. Make sure they are in the same folder as app.py."

model, scaler, load_error = load_model()

if load_error:
    st.error(load_error)
    st.stop()

# ─────────────────────────────────────────
#  GET FEATURE NAMES FROM SCALER
# ─────────────────────────────────────────
try:
    feature_names = scaler.feature_names_in_.tolist()
except AttributeError:
    feature_names = None

# ─────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🏁 Driver & Team</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    driver = st.selectbox("Driver", ["Verstappen", "Hamilton", "Leclerc", "Norris", "Sainz", "Perez"])
with col2:
    team = st.selectbox("Team", ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston Martin", "Alpine"])

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📊 Race Positions</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    quali_pos = st.number_input("Qualifying Position", min_value=1, max_value=20, value=3)
with col4:
    start_pos = st.number_input("Start Position", min_value=1, max_value=20, value=3)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">⛅ Race Conditions</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    weather = st.selectbox("Weather", ["Dry", "Wet"])
with col6:
    tire = st.selectbox("Tire Strategy", ["Soft", "Medium", "Hard"])

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────
if st.button("⚡ PREDICT PODIUM FINISH"):

    # Build raw row exactly as training did
    raw = {
        "Driver":             driver,
        "Team":               team,
        "StartPosition":      start_pos,
        "QualifyingPosition": quali_pos,
        "Weather":            weather,
        "TireStrategy":       tire,
    }
    raw_df = pd.DataFrame([raw])

    # One-hot encode (same as pd.get_dummies in training)
    encoded = pd.get_dummies(raw_df)

    # Align columns to training feature names
    if feature_names:
        encoded = encoded.reindex(columns=feature_names, fill_value=0)
    else:
        st.warning("⚠️ Could not verify feature alignment. Results may be inaccurate.")

    # Scale
    try:
        scaled = scaler.transform(encoded)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # Predict
    try:
        pred       = model.predict(scaled)[0]
        prob       = model.predict_proba(scaled)[0]
        confidence = round(max(prob) * 100, 1)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # ── Result Display ──
    st.markdown("---")
    if pred == 1:
        st.markdown(f"""
        <div class="result-podium">
            <div class="result-title" style="color:#ffd700;">🏆 PODIUM FINISH LIKELY</div>
            <div class="result-sub">Confidence: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="result-no-podium">
            <div class="result-title" style="color:#aaa;">❌ NO PODIUM PREDICTED</div>
            <div class="result-sub">Confidence: {confidence}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🏆 Podium Probability",    f"{round(prob[1]*100, 1)}%")
    with col_b:
        st.metric("❌ No Podium Probability", f"{round(prob[0]*100, 1)}%")
