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
# Config
# =========================

DATA_PATH = "data/Lead Scoring.csv"
MODEL_PATH = "artifacts/lead_scoring_model.pkl"
FEATURES_PATH = "artifacts/selected_features.pkl"
THRESHOLD_PATH = "artifacts/threshold.pkl"

THRESHOLD = 0.4
N_FEATURES = 20

# =========================
# Load Data
# =========================

df = pd.read_csv(DATA_PATH)

# =========================
# Data Cleaning
# =========================

initial_drop = [
    "Prospect ID",
    "Lead Number",
    "I agree to pay the amount through cheque",
]

leakage_drop = [
    col for col in df.columns
    if "Tags" in col
    or "Last Activity" in col
    or "Last Notable Activity" in col
    or "Lead Quality" in col
]

df = df.drop(columns=initial_drop + leakage_drop)

# =========================
# Missing Values
# =========================

df["Asymmetrique Activity Score"] = df["Asymmetrique Activity Score"].fillna("Unknown")
df["Asymmetrique Profile Score"] = df["Asymmetrique Profile Score"].fillna("Unknown")

df["Page Views Per Visit"] = df["Page Views Per Visit"].fillna(df["Page Views Per Visit"].median())
df["TotalVisits_missing"] = df["TotalVisits"].isnull().astype(int)
df["TotalVisits"] = df["TotalVisits"].fillna(df["TotalVisits"].median())

# =========================
# Encoding
# =========================

df = pd.get_dummies(df)

# =========================
# Features & Target
# =========================

X = df.drop(columns=["Converted"])
y = df["Converted"]

# =========================
# Feature Selection with RFE
# =========================

base_model = GradientBoostingClassifier(random_state=42)

rfe = RFE(
    estimator=base_model,
    n_features_to_select=N_FEATURES
)

rfe.fit(X, y)
selected_features = list(X.columns[rfe.support_])
X = X[selected_features]

# =========================
# Evaluation Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Train Evaluation Model
# =========================

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= THRESHOLD).astype(int)

print("Selected Features:")
print(selected_features)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# Final Training for Deployment
# =========================

final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X, y)

# =========================
# Save Artifacts
# =========================

joblib.dump(final_model, MODEL_PATH)
joblib.dump(selected_features, FEATURES_PATH)
joblib.dump(THRESHOLD, THRESHOLD_PATH)

print("\nArtifacts saved successfully.")