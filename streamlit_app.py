import streamlit as st
import pandas as pd
import pickle

# -------- LOAD MODEL --------
model = pickle.load(open("Notebook/model.pkl", "rb"))
scaler = pickle.load(open("Notebook/scaler.pkl", "rb"))

# -------- UI --------
st.title("F1 AI Predictor")
st.write("Predict Formula One Podium Finish")

# -------- INPUTS --------
driver = st.selectbox("Driver", ["Verstappen", "Hamilton", "Leclerc"])
team = st.selectbox("Team", ["Red Bull", "Mercedes", "Ferrari"])

start_pos = st.number_input("Start Position", min_value=1, max_value=20, value=5)
quali_pos = st.number_input("Qualifying Position", min_value=1, max_value=20, value=5)

weather = st.selectbox("Weather", ["Dry", "Wet"])
tire = st.selectbox("Tire Strategy", ["Soft", "Medium", "Hard"])

# -------- SIMPLE ENCODING --------
driver_map = {"Verstappen": 0, "Hamilton": 1, "Leclerc": 2}
team_map = {"Red Bull": 0, "Mercedes": 1, "Ferrari": 2}
weather_map = {"Dry": 0, "Wet": 1}
tire_map = {"Soft": 0, "Medium": 1, "Hard": 2}

# -------- PREDICTION --------
if st.button("Predict"):

    data = [[
        driver_map[driver],
        team_map[team],
        start_pos,
        quali_pos,
        weather_map[weather],
        tire_map[tire]
    ]]

    df = pd.DataFrame(data, columns=[
        "Driver",
        "Team",
        "StartPosition",
        "QualifyingPosition",
        "Weather",
        "TireStrategy"
    ])

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred = model.predict(df_scaled)

    # Output
    if pred[0] == 1:
        st.success("Podium Finish Likely!")
    else:
        st.error("No Podium Finish")