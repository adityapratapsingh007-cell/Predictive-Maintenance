import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="centered"
)

try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

MODEL_PATH = os.path.join(base_dir, "model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(
        base_dir, "predictive_maintenance", "deployment", "model.joblib"
    )

model = joblib.load(MODEL_PATH)

st.title("🔧 Predictive Maintenance")
st.write("Enter engine sensor readings to predict the engine condition.")

Engine_rpm = st.number_input("Engine RPM", min_value=0.0, value=700.0)
Lub_oil_pressure = st.number_input("Lub Oil Pressure", min_value=0.0, value=3.3)
Fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, value=6.7)
Coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, value=2.3)
lub_oil_temp = st.number_input("Lub Oil Temperature", min_value=0.0, value=77.6)
Coolant_temp = st.number_input("Coolant Temperature", min_value=0.0, value=78.4)

input_data = pd.DataFrame([{
    "Engine rpm": Engine_rpm,
    "Lub oil pressure": Lub_oil_pressure,
    "Fuel pressure": Fuel_pressure,
    "Coolant pressure": Coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": Coolant_temp,
}])

if st.button("Predict Engine Condition"):
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        st.warning("⚠️ Maintenance / Failure-Risk Condition Detected")
    else:
        st.success("✅ Engine Operating in Normal Condition")

    st.metric("Maintenance-Risk Probability", f"{probability:.2%}")
    st.dataframe(input_data, use_container_width=True)
