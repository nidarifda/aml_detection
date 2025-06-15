import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === Load the scaler ===
scaler_path = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = joblib.load(f)

# === Define the features used in training ===
FEATURE_COLUMNS = [
    'time_since_last', 'tx_count', 'total_paid', 'is_dual_role', 'amount_volatility',
    'structuring_flag', 'hour_deviation', 'high_freq_sender', 'high_risk_currency',
    'hour_sin', 'hour_cos', 'is_weekend',
    'sender_degree', 'receiver_degree', 'activity_ratio', 'rapid_sequence',
    'structured_tx', 'component_id', 'sent_to', 'received_from'
]

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the input dataframe using only the selected features and applies the trained scaler.
    """
    df_clean = df.copy()

    # Ensure all expected columns are present
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for prediction: {missing_cols}")

    # Select and reorder columns
    df_clean = df_clean[FEATURE_COLUMNS]

    # Handle missing values (if any)
    df_clean.fillna(0, inplace=True)

    # Apply scaling
    df_scaled = scaler.transform(df_clean)

    return pd.DataFrame(df_scaled, columns=FEATURE_COLUMNS)
