<img width="1814" height="886" alt="image" src="https://github.com/user-attachments/assets/3175475e-30c1-4ea8-b6fd-0209f9b1ed41" />


# 🎯 Explainable AI Lead Scoring Dashboard

An end-to-end machine learning project that predicts lead conversion probability and provides explainable AI insights for business decision-making.

This project was built to simulate a real-world sales prioritization tool — from data cleaning and feature engineering to deployment-ready prediction pipelines.

---

## 🚀 Features

### Machine Learning
- Data cleaning & preprocessing
- Missing value handling
- Leakage detection & removal
- Feature selection (RFE)
- Threshold optimization
- Gradient Boosting model
- Explainable AI using SHAP
- Probability-based lead scoring

### Dashboard
- Interactive Streamlit app
- Single lead prediction
- Batch CSV upload
- CSV export
- Adjustable decision threshold
- Hot / Warm / Cold lead categorization
- Explainability (Why this score?)

### ML System Components
- Saved model artifacts (.pkl)
- Modular prediction pipeline
- Feedback loop collection
- Deployment-ready project structure

---

## 📊 Example Workflow

Lead Data

↓

Prediction Score (0–100)

↓

Business Category

```text
Hot Lead
Warm Lead
Cold Lead
```

↓

SHAP Explanation

↓

Feedback Collection

↓

Future Retraining

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- SHAP
- Streamlit
- Joblib
- Git / GitHub

---

## 📁 Project Structure

```text
Lead Scoring/
│
├── app.py                 # Streamlit dashboard
├── model_utils.py         # Prediction pipeline
├── train_model.py         # Model training
├── artifacts/             # Saved model files
├── feedback_data.csv      # Feedback loop data
├── requirements.txt
└── README.md
```

---

## 🎯 Goal

The goal of this project is to move beyond simple ML prediction and build a small explainable AI system with deployment-ready components and business value.

---

## 🌐 Live Demo

https://lead-scoring-dashboard-vh8byhndruuhyvuqkwkjrq.streamlit.app/

```
Streamlit deployment link here
```

---

Built by Sascha
