import streamlit as st
import numpy as np
import pickle

st.title("🎓 Jamboree Admission Predictor")

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

gre = st.number_input("GRE Score", 260, 340, 300)
toefl = st.number_input("TOEFL Score", 0, 120, 100)
university_rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("SOP", 1.0, 5.0, 3.0, 0.5)
lor = st.slider("LOR", 1.0, 5.0, 3.0, 0.5)
cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
research = st.selectbox("Research", [0, 1])

if st.button("Predict"):
    overall_score = gre + toefl
    sop_lor_interaction = sop * lor

    input_data = np.array([[gre, toefl, university_rating, sop, lor,
                            cgpa, research, overall_score, sop_lor_interaction]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prediction = max(0, min(1, prediction))

    st.success(f"Admission Chance: {prediction:.2f}")

    if prediction >= 0.75:
        st.write("✅ High Chance")
    elif prediction >= 0.50:
        st.write("⚠️ Medium Chance")
    else:
        st.write("❌ Low Chance")
