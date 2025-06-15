import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from utils.preprocessing import preprocess_input, expected_features

# === Page Config ===
st.set_page_config(page_title="AML Detection - DNN Model", layout="centered")

# === Load Model & Scaler ===
model = tf.keras.models.load_model("best_model.h5")

st.title("Anti-Money Laundering Detection (DNN Model)")
st.markdown("Enter the transaction details below to predict the risk of money laundering.")

# === Input Widgets ===
user_input = {}
with st.sidebar:
    st.header("Transaction Input")

    user_input['time_since_last'] = st.slider("Time Since Last Tx (hrs)", 0, 168, 24)
    user_input['tx_count'] = st.slider("Transaction Count", 1, 100, 5)
    user_input['total_paid'] = st.number_input("Total Paid (USD)", 0.0, 1e6, 1000.0)
    user_input['is_dual_role'] = st.selectbox("Is Dual Role?", [0, 1])
    user_input['amount_volatility'] = st.slider("Amount Volatility", 0.0, 1.0, 0.5)
    user_input['structuring_flag'] = st.selectbox("Structuring Detected?", [0, 1])
    user_input['hour_deviation'] = st.slider("Hour Deviation", 0.0, 5.0, 1.0)
    user_input['high_freq_sender'] = st.selectbox("High Frequency Sender?", [0, 1])
    user_input['high_risk_currency'] = st.selectbox("High Risk Currency?", [0, 1])
    user_input['hour_sin'] = st.slider("Hour (sin)", -1.0, 1.0, 0.0)
    user_input['hour_cos'] = st.slider("Hour (cos)", -1.0, 1.0, 1.0)
    user_input['is_weekend'] = st.selectbox("Weekend Transaction?", [0, 1])
    user_input['sender_degree'] = st.slider("Sender Degree", 0, 100, 10)
    user_input['receiver_degree'] = st.slider("Receiver Degree", 0, 100, 10)
    user_input['activity_ratio'] = st.slider("Activity Ratio", 0.0, 1.0, 0.5)
    user_input['rapid_sequence'] = st.selectbox("Rapid Sequence?", [0, 1])
    user_input['structured_tx'] = st.selectbox("Structured Transaction?", [0, 1])
    user_input['component_id'] = st.slider("Graph Component ID", 0, 10000, 100)
    user_input['sent_to'] = st.slider("Sent To (Accounts)", 0, 500, 10)
    user_input['received_from'] = st.slider("Received From (Accounts)", 0, 500, 10)

# === Prediction ===
if st.button("Predict Laundering Risk"):
    try:
        X = preprocess_input(user_input)
        prediction = model.predict(X)[0][0]
        label = "Likely Laundering" if prediction > 0.5 else "Legitimate Transaction"

        st.subheader("Prediction Result")
        st.metric(label="Prediction", value=label)
        st.progress(prediction)

        st.write(f"**Probability of laundering:** `{prediction:.2%}`")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
