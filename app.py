import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---- Symptom Categories ----
symptom_categories = {
    "Mood": ["anxiety", "depressed_mood"],
    "Cognitive": ["memory_loss", "paranoia"],
    "Perceptual": ["hallucinations", "delusions"],
    "Behavioral": ["impulsivity", "sleep_disturbance"]
}

# ---- Dataset ----
data = {
    "hallucinations": [1, 0, 0, 0, 0],
    "delusions": [1, 1, 0, 0, 0],
    "anxiety": [0, 0, 1, 0, 1],
    "depressed_mood": [0, 0, 1, 1, 0],
    "impulsivity": [0, 0, 0, 1, 0],
    "memory_loss": [0, 0, 0, 0, 1],
    "sleep_disturbance": [0, 0, 1, 1, 1],
    "paranoia": [1, 1, 0, 0, 1],
    "diagnosis": ["schizophrenia", "bipolar_disorder", "ptsd", "major_depression", "borderline_personality"]
}
df = pd.DataFrame(data)
symptoms = list(symptom_categories.values())
symptoms = [item for sublist in symptoms for item in sublist]

X = df[symptoms]
y = df["diagnosis"]
model = RandomForestClassifier()
model.fit(X, y)

# ---- Streamlit UI ----
st.set_page_config(page_title="🧠 Mental Health Diagnostic Assistant", page_icon="🧠")
st.title("🧠 Mental Health Diagnostic Assistant")

# ---- Filter symptoms ----
st.sidebar.header("🔍 Filter Symptoms by Category")
selected_categories = st.sidebar.multiselect("Select categories", symptom_categories.keys())

filtered_symptoms = []
for cat in selected_categories:
    filtered_symptoms += symptom_categories[cat]

selected_symptoms = st.multiselect("Select your symptoms:", filtered_symptoms)

# ---- Severity ----
st.markdown("### 🔥 Symptom Severity")
severity_levels = {"Mild": 1, "Moderate": 2, "Severe": 3}
user_input = []

for symptom in symptoms:
    if symptom in selected_symptoms:
        severity = st.selectbox(f"Severity of {symptom}", list(severity_levels.keys()), key=symptom)
        user_input.append(severity_levels[severity])
    else:
        user_input.append(0)

# ---- Prediction ----
if any(user_input):
    proba = model.predict_proba([user_input])[0]
    labels = model.classes_

    st.markdown("### 📊 Diagnosis Probabilities")
    for label, prob in zip(labels, proba):
        st.progress(prob)
        st.write(f"**{label}**: {prob:.2f}")
    
    # Red flag warning
    if user_input[symptoms.index("hallucinations")] > 0 and user_input[symptoms.index("paranoia")] > 0:
        st.error("⚠️ Urgent Attention Recommended: Combination of hallucinations and paranoia may indicate a serious condition. Please consult a mental health professional.")

else:
    st.info("Select at least one symptom to begin.")

st.caption("🧪 Prototype — Not a diagnostic tool. Please seek professional evaluation for any mental health concerns.")
