# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("student_health_data.csv")
    df = df.drop(columns=["Student_ID"])
    df['Physical_Activity'] = df['Physical_Activity'].map({'Low': 1, 'Moderate': 2, 'High': 3})
    df['Sleep_Quality'] = df['Sleep_Quality'].map({'Poor': 1, 'Moderate': 2, 'Good': 3})
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 2})
    df['Mood'] = df['Mood'].map({'Stressed': 1, 'Neutral': 2, 'Happy': 3})
    df['Health_Risk_Level'] = df['Health_Risk_Level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
    return df

df = load_data()
X = df.drop(columns=["Health_Risk_Level"])
y = df["Health_Risk_Level"]

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# UI Toggle
st.title("Student Health Risk App")
mode = st.radio("Choose Mode:", ["Predict Risk", "Optimize Health"])

# Mapping dictionaries
gender_map = {"M": 1, "F": 2}
activity_map = {"Low": 1, "Moderate": 2, "High": 3}
sleep_map = {"Poor": 1, "Moderate": 2, "Good": 3}
mood_map = {"Stressed": 1, "Neutral": 2, "Happy": 3}

if mode == "Predict Risk":
    st.write("Enter your information to predict your health risk level:")

    gender = st.selectbox("Gender", list(gender_map.keys()))
    age = st.slider("Age", 10, 30)
    heart_rate = st.slider("Heart Rate", 40, 120, 70)
    bp_systolic = st.slider("Systolic Blood Pressure", 90, 180, 120)
    bp_diastolic = st.slider("Diastolic Blood Pressure", 60, 120, 80)
    stress_biosensor = st.slider("Stress Level (Biosensor)", 0, 100, 50)
    stress_self = st.slider("Stress Level (Self Report)", 0, 10, 5)
    physical_activity = st.selectbox("Physical Activity Level", list(activity_map.keys()))
    sleep_quality = st.selectbox("Sleep Quality", list(sleep_map.keys()))
    mood = st.selectbox("Mood", list(mood_map.keys()))
    study_hours = st.slider("Study Hours Per Day", 0, 12, 4)
    project_hours = st.slider("Project Hours Per Day", 0, 12, 4)

    if st.button("Predict Health Risk"):
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender_map[gender],
            "Heart_Rate": heart_rate,
            "Blood_Pressure_Systolic": bp_systolic,
            "Blood_Pressure_Diastolic": bp_diastolic,
            "Stress_Level_Biosensor": stress_biosensor,
            "Stress_Level_Self_Report": stress_self,
            "Physical_Activity": activity_map[physical_activity],
            "Sleep_Quality": sleep_map[sleep_quality],
            "Mood": mood_map[mood],
            "Study_Hours": study_hours,
            "Project_Hours": project_hours
        }])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        risk_label = {1: "Low", 2: "Moderate", 3: "High"}

        st.subheader("Predicted Health Risk Level:")
        rounded_prediction = int(np.clip(round(prediction), 1, 3))
        st.success(risk_label[rounded_prediction])

elif mode == "Optimize Health":
    st.write("Input any known values. Leave others blank or set to 'Optimize' to see suggestions for improving health.")

    feature_defaults = X.mean().to_dict()
    optim_inputs = {}
    locked_features = {}

    reverse_maps = {
        "Gender": gender_map,
        "Physical_Activity": activity_map,
        "Sleep_Quality": sleep_map,
        "Mood": mood_map
    }
    reverse_maps_inv = {col: {v: k for k, v in mapping.items()} for col, mapping in reverse_maps.items()}

    for col in X.columns:
        if col in reverse_maps:
            selected = st.selectbox(f"{col} (Leave blank to optimize)", ["Optimize"] + list(reverse_maps[col].keys()))
            if selected != "Optimize":
                val = reverse_maps[col][selected]
                optim_inputs[col] = val
                locked_features[col] = val
        else:
            val = st.text_input(f"{col} (Leave blank to optimize)", value="")
            if val.strip() != "":
                try:
                    val = float(val)
                    optim_inputs[col] = val
                    locked_features[col] = val
                except:
                    st.warning(f"Invalid input for {col}, skipping...")

    if st.button("Suggest Health Improvements"):
        current = feature_defaults.copy()
        current.update(optim_inputs)

        best_input = None
        best_score = float('inf')

        for _ in range(500):
            trial = current.copy()
            for col in X.columns:
                if col not in locked_features:
                    trial[col] = np.random.normal(loc=feature_defaults[col], scale=X[col].std())
            trial_scaled = scaler.transform(pd.DataFrame([trial]))
            score = model.predict(trial_scaled)[0]
            if score < best_score:
                best_score = score
                best_input = trial.copy()

        st.subheader("Suggested Changes for Optimal Health")
        for col in X.columns:
            if col not in locked_features:
                change = best_input[col] - current[col]
                suggestion = best_input[col]
                if col in reverse_maps_inv:
                    suggestion_rounded = int(np.clip(round(suggestion), 1, 3))
                    label = reverse_maps_inv[col].get(suggestion_rounded, "Unknown")
                    st.write(f"**{col}**: Change to '{label}'")
                else:
                    st.write(f"**{col}**: Change to {round(suggestion, 2)} (Δ {round(change, 2)})")

        final_risk = int(np.clip(round(best_score), 1, 3))
        st.success(f"Estimated Health Risk Level: {final_risk}")

        # --- Lifestyle Suggestions ---
        st.subheader("Lifestyle Suggestions")
        for col in X.columns:
            if col not in locked_features:
                if col == "Physical_Activity":
                    if best_input[col] < 2:
                        st.info("Consider incorporating more physical activity like daily walks, gym sessions, or sports.")
                    st.write(f"**{col}**: Try to reach at least a 'Moderate' activity level for better cardiovascular and metabolic health.")
                elif col == "Sleep_Quality":
                    if best_input[col] < 2:
                        st.info("Try to improve your sleep by maintaining a regular sleep schedule and reducing screen time before bed.")
                    st.write(f"**{col}**: Aim for 'Moderate' to 'Good' sleep quality to support recovery and mental well-being.")
                elif col == "Mood":
                    if best_input[col] < 2:
                        st.info("Engage in mindfulness practices, social activities, or seek counseling to support emotional well-being.")
                    st.write(f"**{col}**: Improved mood is linked to lower stress and better academic performance.")
                elif col == "Blood_Pressure_Systolic":
                    if best_input[col] > 130:
                        st.info("High blood pressure detected. Consider reducing sodium intake, exercising regularly, and consulting a doctor.")
                    elif best_input[col] < 90:
                        st.info("Low blood pressure detected. Stay hydrated, eat smaller meals more often, and consult a healthcare provider.")
                    st.write("**Blood_Pressure_Systolic**: Ideal range is 90-120. Aim to stay within this range for heart health.")
                elif col == "Blood_Pressure_Diastolic":
                    if best_input[col] > 90:
                        st.info("Elevated diastolic pressure. Consider lifestyle changes like reducing caffeine and managing stress.")
                    elif best_input[col] < 60:
                        st.info("Low diastolic pressure may cause fatigue. Stay hydrated and avoid standing up too quickly.")
                    st.write("**Blood_Pressure_Diastolic**: Ideal range is 60-80.")
                elif col == "Stress_Level_Biosensor":
                    if best_input[col] > 60:
                        st.info("High stress levels detected. Consider stress-relief techniques like deep breathing, breaks during work, or yoga.")
                    st.write("**Stress_Level_Biosensor**: Lower readings suggest better autonomic balance and stress management.")
                elif col == "Stress_Level_Self_Report":
                    if best_input[col] > 6:
                        st.info("Self-reported stress is high. Try journaling or setting daily mental check-ins.")
                    st.write("**Stress_Level_Self_Report**: Lower stress helps improve focus and health.")
                elif col == "Heart_Rate":
                    st.write("**Heart_Rate**: A resting heart rate of 60–80 bpm is generally healthy.")
                elif col == "Age":
                    st.write("**Age**: While age is not adjustable, younger individuals generally recover faster. Maintain good habits early.")
                elif col == "Study_Hours":
                    st.write("**Study_Hours**: Balanced study time (3–5 hours) supports learning without burning out.")
                elif col == "Project_Hours":
                    st.write("**Project_Hours**: Avoid long continuous hours—break tasks into chunks for mental wellness.")
                elif col == "Gender":
                    st.write("**Gender**: Gender itself doesn't affect risk, but some health markers may vary slightly by sex.")

