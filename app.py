import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# === Load Model and Preprocessing Artifacts ===
@st.cache_resource
def load_artifacts():
    model = load_model("dnn_aml_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    try:
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        threshold = metadata.get("threshold", 0.5)
        feature_names = metadata.get("features", [])
    except:
        st.warning("Metadata not found. Using default threshold (0.5) and auto-selecting numeric features.")
        threshold = 0.5
        feature_names = []
    return model, scaler, threshold, feature_names

model, scaler, threshold, feature_names = load_artifacts()

# === Streamlit Page Configuration ===
st.set_page_config(page_title="AML Transaction Classifier", layout="wide")
st.title("Anti-Money Laundering (AML) Detection System")

st.markdown("""
This application deploys a Deep Neural Network (DNN) model trained on synthetic financial transactions to flag potential laundering behavior. 
You can either upload a CSV file or manually enter values to test the model's predictions.
""")

# === Prediction Function ===
def predict(input_df):
    scaled_input = scaler.transform(input_df)
    probabilities = model.predict(scaled_input).flatten()
    predictions = (probabilities >= threshold).astype(int)
    results = input_df.copy()
    results["Fraud Probability"] = np.round(probabilities, 4)
    results["AML Risk Classification"] = np.where(predictions == 1, "⚠️ Suspicious", "✅ Legitimate")
    return results

# === Sidebar Input Method Selection ===
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Select how to provide transaction data:", ["Upload CSV File", "Manual Input"])

# === Upload CSV File ===
if input_method == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            input_features = df[feature_names] if feature_names else df.select_dtypes(include=[np.number])
            if input_features.empty:
                st.error("No numeric features available for prediction.")
            else:
                output_df = predict(input_features)
                st.success(f"Processed {len(df)} transactions.")
                st.dataframe(output_df)

                st.subheader("Summary Report")
                suspicious_count = (output_df["AML Risk Classification"] == "⚠️ Suspicious").sum()
                st.metric(label="Suspicious Transactions", value=suspicious_count)
        except Exception as e:
            st.error(f"File processing failed: {e}")

# === Manual Input ===
elif input_method == "Manual Input":
    if not feature_names:
        st.warning("Manual input disabled. Feature names not found in metadata.")
    else:
        st.sidebar.subheader("Enter Feature Values")
        user_input = {feature: st.sidebar.number_input(f"{feature}", value=0.0) for feature in feature_names}
        input_df = pd.DataFrame([user_input])
        if st.sidebar.button("Classify"):
            result_df = predict(input_df)
            st.subheader("Classification Result")
            st.table(result_df[["Fraud Probability", "AML Risk Classification"]])

# === Footer ===
st.markdown("---")
st.caption("Developed as part of MSc Cybersecurity Final Year Project | AML Detection via Deep Learning & Federated Learning")
