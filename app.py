import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# === Load Model and Preprocessing Artifacts ===
@st.cache_resource
def load_artifacts():
    model = load_model("dnn_aml_model.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    try:
        with open("metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        threshold = metadata.get("threshold", 0.5)
        feature_names = metadata.get("features", [])
    except:
        threshold = 0.5
        feature_names = []
    return model, scaler, threshold, feature_names

model, scaler, threshold, feature_names = load_artifacts()

# === Streamlit Page Configuration ===
st.set_page_config(page_title="AML Transaction Classifier", layout="wide")
st.title("Anti-Money Laundering (AML) Detection System")
st.markdown("""
This application deploys a Deep Neural Network (DNN) model to detect potentially suspicious transactions. 
Users can input financial transaction data either manually or by uploading a CSV file. 
The model evaluates each transaction based on pre-engineered behavioral and structural features.
""")

# === Prediction Function ===
def predict(input_df):
    scaled_input = scaler.transform(input_df)
    probabilities = model.predict(scaled_input).flatten()
    predictions = (probabilities >= threshold).astype(int)
    results = input_df.copy()
    results["Fraud Probability"] = probabilities
    results["AML Risk Classification"] = np.where(predictions == 1, "Suspicious", "Legitimate")
    return results

# === Input Method Selection ===
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose how to provide transaction data:", ["Upload CSV File", "Manual Input"])

# === File Upload ===
if input_method == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with transaction data", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            input_features = df[feature_names] if feature_names else df.select_dtypes(include=[np.number])
            output_df = predict(input_features)
            st.success(f"Processed {len(df)} transactions.")
            st.dataframe(output_df)

            st.subheader("Summary")
            suspicious_count = (output_df["AML Risk Classification"] == "Suspicious").sum()
            st.write(f"Suspicious transactions detected: {suspicious_count}")
        except Exception as e:
            st.error(f"Failed to process the file: {e}")

# === Manual Input Form ===
elif input_method == "Manual Input":
    if not feature_names:
        st.warning("Manual input is unavailable because feature names are missing from metadata.")
    else:
        st.sidebar.subheader("Enter feature values")
        user_input = {feature: st.sidebar.number_input(f"{feature}", value=0.0) for feature in feature_names}
        input_df = pd.DataFrame([user_input])
        if st.sidebar.button("Classify"):
            result_df = predict(input_df)
            st.subheader("Classification Result")
            st.write(result_df[["Fraud Probability", "AML Risk Classification"]])

# === Footer ===
st.markdown("---")
st.caption("Developed for MSc Cybersecurity Final Year Project | AML Detection using Deep Learning and Federated Learning Techniques")
