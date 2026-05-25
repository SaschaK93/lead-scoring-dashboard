# =========================
# Imports
# =========================

import pandas as pd
import joblib
import shap

# =========================
# Load Artifacts
# =========================

MODEL_PATH = "artifacts/lead_scoring_model.pkl"
FEATURES_PATH = "artifacts/selected_features.pkl" # Loading final features after RFE
THRESHOLD_PATH = "artifacts/threshold.pkl" # Loading configured threshold from training (e.g., 0.4)

model = joblib.load(MODEL_PATH)
selected_features = joblib.load(FEATURES_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# =========================
# Prediction Function
# =========================

def predict_lead_score(input_data):

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in selected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only selected features
    input_df = input_df[selected_features]

    # Ensure numeric format
    input_df = input_df.astype(float)

    # Predict probability
    probability = model.predict_proba(input_df)[:, 1][0]

    # Convert to score
    lead_score = round(probability * 100)

    # Apply threshold
    prediction = int(probability >= threshold)

    # Lead category
    if lead_score >= 80:
        lead_category = "Hot Lead"

    elif lead_score >= 50:
        lead_category = "Warm Lead"

    else:
        lead_category = "Cold Lead"

    # SHAP values
    shap_values = explainer.shap_values(input_df)

    # Convert SHAP values to readable feature contributions
    contributions = pd.Series(shap_values[0],index=input_df.columns).sort_values(key=abs, ascending=False)
    top_contributions = contributions.head(5)

    return lead_score, prediction, lead_category, top_contributions
