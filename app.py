import streamlit as st
import numpy as np
import pickle

st.title("🎓 Ivy League Admission Predictor")

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.write("Enter student details:")

# Inputs
gre = st.number_input("GRE Score", 260, 340, 300)
toefl = st.number_input("TOEFL Score", 0, 120, 100)
university_rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, 0.5)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, 0.5)
cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
research = st.selectbox("Research Experience", [0, 1])

# Predict
if st.button("Predict"):

    # 🔥 Derived features (from your project)
    overall_score = (gre/340) + (toefl/120) + (cgpa/10)
    sop_lor_interaction = sop * lor

    # Input (same order as training)
    input_data = np.array([[gre, toefl, university_rating, sop, lor,
                            cgpa, research, overall_score, sop_lor_interaction]])

    prediction = model.predict(input_data)[0]

    st.success(f"Admission Chance: {prediction:.2f}")

    if prediction > 0.8:
        st.write("🎉 Very High Chance")
    elif prediction > 0.6:
        st.write("👍 Good Chance")
    else:
        st.write("⚠️ Low Chance")
