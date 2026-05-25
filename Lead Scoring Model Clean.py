# =========================
# Imports
# =========================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE

# =========================
# Load Data
# =========================

df = pd.read_csv(r'C:\Users\sasch\OneDrive\Desktop\Python\Projekte\Lead Scoring\Lead Scoring.csv')

# =========================
# Data Cleaning
# =========================

initial_drop = ['Prospect ID','Lead Number','I agree to pay the amount through cheque']
leakage_drop = [
    col for col in df.columns
    if 'Tags' in col
    or 'Last Activity' in col
    or 'Last Notable Activity' in col
    or 'Lead Quality' in col
]
df = df.drop(columns=initial_drop + leakage_drop)

# =========================
# Missing Values
# =========================

df['Asymmetrique Activity Score'] = (df['Asymmetrique Activity Score'].fillna('Unknown'))
df['Asymmetrique Profile Score'] = (df['Asymmetrique Profile Score'].fillna('Unknown'))
df['Page Views Per Visit'] = (df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median()))
df['TotalVisits_missing'] = (df['TotalVisits'].isnull().astype(int))
df['TotalVisits'] = (df['TotalVisits'].fillna(df['TotalVisits'].median()))

# =========================
# Encoding
# =========================

df = pd.get_dummies(df, drop_first=True)

# =========================
# Features & Target
# =========================

X = df.drop(columns=['Converted'])
y = df['Converted']

# =========================
# Recursive Feature Elimination (RFE)
# =========================

initial_model = GradientBoostingClassifier(random_state=42)
rfe = RFE(estimator=initial_model,n_features_to_select=20)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
X = X[selected_features]

print("Selected Features:")
print(list(selected_features))

# =========================
# Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# =========================
# Final Model
# =========================

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# =========================
# Predictions
# =========================

THRESHOLD = 0.4
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

# =========================
# Evaluation
# =========================

print(classification_report(y_test, y_pred))

# =========================
# Final Training for Deployment
# =========================

final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X, y)

# =========================
# Save Model Artifacts
# =========================

joblib.dump(final_model, "lead_scoring_model.pkl")
joblib.dump(list(selected_features), "selected_features.pkl")
joblib.dump(THRESHOLD, "threshold.pkl")

# =========================
# Prediction Function
# =========================

import joblib

model = joblib.load("lead_scoring_model.pkl")
selected_features = joblib.load("selected_features.pkl")
threshold = joblib.load("threshold.pkl")


def predict_lead_score(input_data):
    """
    input_data: dictionary with raw lead information
    returns: lead score, prediction, lead category
    """

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode input
    input_df = pd.get_dummies(input_df)

    # Add missing selected feature columns
    for col in selected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only selected features in correct order
    input_df = input_df[selected_features]

    # Ensure numeric format
    input_df = input_df.astype(float)

    # Predict probability
    probability = model.predict_proba(input_df)[:, 1][0]

    # Convert to score
    lead_score = round(probability * 100)

    # Apply threshold
    prediction = int(probability >= threshold)

    # Business category
    if lead_score >= 80:
        lead_category = "Hot Lead"
    elif lead_score >= 50:
        lead_category = "Warm Lead"
    else:
        lead_category = "Cold Lead"

    return lead_score, prediction, lead_category

# =========================
# Test Prediction Function
# =========================

test_lead = {
    "Total Time Spent on Website": 800,
    "TotalVisits": 6,
    "Page Views Per Visit": 3.5,
    "TotalVisits_missing": 0,

    "Lead Origin": "Lead Add Form", 
    "Do Not Email": "No",
    "Specialization": "Select",
    "How did you hear about X Education": "Select",
    "What is your current occupation": "Working Professional",
    "Lead Profile": "Potential Lead",
    "City": "Select",
    "Asymmetrique Activity Index": "03.Low",
    "Asymmetrique Activity Score": 15.0,
    "Asymmetrique Profile Score": "Unknown"
}

score, prediction, category = predict_lead_score(test_lead)

print("Lead Score:", score)
print("Prediction:", prediction)
print("Category:", category)

cold_lead = {
    "Total Time Spent on Website": 20,
    "TotalVisits": 1,
    "Page Views Per Visit": 1.0,
    "TotalVisits_missing": 0,

    "Lead Origin": "Landing Page Submission",
    "Do Not Email": "Yes",
    "Specialization": "Select",
    "How did you hear about X Education": "Select",
    "What is your current occupation": "Unemployed",
    "Lead Profile": "Select",
    "City": "Select",
    "Asymmetrique Activity Index": "03.Low",
    "Asymmetrique Activity Score": 13.0,
    "Asymmetrique Profile Score": "Unknown"
}

hot_lead = {
    "Total Time Spent on Website": 1200,
    "TotalVisits": 8,
    "Page Views Per Visit": 4.5,
    "TotalVisits_missing": 0,

    "Lead Origin": "Lead Add Form",
    "Do Not Email": "No",
    "Specialization": "Select",
    "How did you hear about X Education": "Select",
    "What is your current occupation": "Working Professional",
    "Lead Profile": "Potential Lead",
    "City": "Select",
    "Asymmetrique Activity Index": "03.Low",
    "Asymmetrique Activity Score": 15.0,
    "Asymmetrique Profile Score": "Unknown"
}

for name, lead in {
    "Cold Lead Test": cold_lead,
    "Warm Lead Test": test_lead,
    "Hot Lead Test": hot_lead
}.items():
    score, prediction, category = predict_lead_score(lead)
    print(name)
    print("Score:", score)
    print("Prediction:", prediction)
    print("Category:", category)
    print("-----")