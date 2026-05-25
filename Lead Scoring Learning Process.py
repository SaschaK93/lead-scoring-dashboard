# Import necessary libraries
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('C:\\Users\\sasch\\OneDrive\\Desktop\\Python\\Projekte\\Lead Scoring\\Lead Scoring.csv')

# Obvious Cleanup: Dropping columns that are not useful for modeling and may cause data leakage after 1st check
initial_drop = ['Prospect ID', 'Lead Number', 'I agree to pay the amount through cheque']

# Deeper Analysis Cleanup: Dropping features identified as causing data leakage based on further feature importance and correlation analysis
leakage_drop = [col for col in df.columns if 
                'Tags' in col or
                'Last Activity' in col or
                'Last Notable Activity' in col or
                'Lead Quality' in col]

df = df.drop(columns=initial_drop + leakage_drop)

# Handling missing values for categorical variables by filling them with 'Unknown'
df['Asymmetrique Activity Score'] = df['Asymmetrique Activity Score'].fillna('Unknown')
df['Asymmetrique Profile Score'] = df['Asymmetrique Profile Score'].fillna('Unknown')

# Handling missing values for numerical variables by filling them with the median
df['Page Views Per Visit'] = df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median())

# Missing Indicator: Create a new binary column to indicate whether 'TotalVisits' was missing for model to capture potential information from missingness
df['TotalVisits_missing'] = df['TotalVisits'].isnull().astype(int)

# Numerical → Median
df['TotalVisits'] = df['TotalVisits'].fillna(df['TotalVisits'].median())

# Encoding categorical variables using one-hot encoding
df = pd.get_dummies(df, drop_first=True) 

# Train simple model to check for data leakage and get feature importance

# 1. X / y erstellen
X = df.drop(columns=['Converted'])
y = df['Converted']

# 2. Train/Test Split für Evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Full model nur auf Trainingsdaten trainieren
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import classification_report
gb_full = GradientBoostingClassifier(random_state=42)
gb_full.fit(X_train, y_train)

# 4. Feature Importance aus Trainingsmodell holen
feature_importance = pd.Series(
    gb_full.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

important_features = feature_importance[feature_importance > 0.005].index

# 5. Reduced Features bauen
X_reduced = X[important_features]

# 6. Neuer Split mit reduziertem X
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# 7. Reduced model evaluieren
gb_reduced = GradientBoostingClassifier(random_state=42)
gb_reduced.fit(X_train_r, y_train_r)

y_proba = gb_reduced.predict_proba(X_test_r)[:, 1]
threshold = 0.4
y_pred = (y_proba >= threshold).astype(int)

print(classification_report(y_test_r, y_pred))

# Train final model for deployment
final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X_reduced, y)

print("FIN")

'''

--------------------- Initial Data Exploration ---------------------
# Display the first few rows of the dataset
print(df.head())

# Check the shape of the dataset
print(df.shape)

# Column names in the dataset
print(df.columns)

# Information about the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Analyze the target variable 'Converted'
print(df['Converted'].value_counts(normalize=True))

Conclusion:
The dataset has a moderate class imbalance (~38% conversion), which will need to be addressed during model training. (5%-20% is typical for lead scoring datasets, so this is within a reasonable range.)
Accuracy alone would be misleading, so I plan to focus on precision-recall trade-offs.
Early exploration suggests that lead source, engagement signals, and user background are likely key drivers of conversion.
The dataset also contains substantial missing values, which will require careful preprocessing.
The asymmetric features appear to be pre-engineered scores. We should validate how they are generated to avoid potential data leakage.”
The I agree to pay column has a high chance of being close to conversion, so it needs to be handled carefully to avoid data leakage."

# Analyze missing values as a percentage of total entries
print('Missing values percentage:')
print(df.isnull().mean().sort_values(ascending=False))

print('Data types distribution:')
print(df.dtypes.value_counts())

print('Top 10 columns with most missing values:')
print(df.isnull().sum().sort_values(ascending=False).head(10))


Initially dropped columns:
- 'Lead Number': Unique identifier, not useful for modeling.
- 'Prospect ID': Another unique identifier, redundant with 'Lead Number'.
- 'I agree to pay': High correlation with conversion, likely to cause data leakage.
--------------------- Initial Data Exploration ---------------------


--------------------- Confusion Matrix and Classification Report ---------------------
Initial Confusion Matrix and Classification Report:

              precision    recall  f1-score   support

           0       0.94      0.96      0.95      1107
           1       0.94      0.90      0.92       741

    accuracy                           0.94      1848
   macro avg       0.94      0.93      0.94      1848
weighted avg       0.94      0.94      0.94      1848

The values seem to be too good, which raises concerns about potential data leakage. 
The model might be picking up on patterns that are not generalizable to new data. 
This could be due to the presence of features that are highly correlated with the target variable, 
or it could indicate that the model is overfitting to the training data. 
Further investigation is needed to identify and address any sources of data leakage before deploying the model.

Threshold	Precision	Recall	Interpretation
0.3	        0.90	    0.94	many leads, but lower quality
0.5	        0.94	    0.90	good balance between quantity and quality
0.7	        0.96	    0.85	fewer leads, but higher quality
0.9	        0.98	    0.74	very few leads, but very high quality

By adjusting the decision threshold, we can control the trade-off between lead quality and coverage. 
For example, at a threshold of 0.7, we achieve 96% precision while still capturing 85% of conversions.
--------------------- Confusion Matrix and Classification Report ---------------------


--------------------- Data Leakage Analysis ---------------------
Accuracy of 94% is suspiciously high, so we check for data leakage by checking the correlation of features as well as feature importance from a Random Forest model.
Checking the correlation of features with the target variable 'Converted', near 1 or -1 are indicators for data leakage.
print("Correlation of features with the target variable 'Converted':")
print(df.corr()['Converted'].sort_values(ascending=False).head(10))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

import pandas as pd
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).head(10)
print("Top 10 most important features according to Random Forest:")
print(feature_importance.sort_values(ascending=False).head(10))

RESULTS:

Top 10 most important features according to Random Forest:
Tags_Will revert after reading the email    0.160635
Total Time Spent on Website                 0.113508
Tags_Ringing                                0.051225
Last Notable Activity_SMS Sent              0.041927
Lead Profile_Potential Lead                 0.041807
Tags_Closed by Horizzon                     0.032587
Lead Quality_Might be                       0.032376
Last Activity_SMS Sent                      0.030135
Tags_Lost to EINS                           0.024837
Page Views Per Visit                        0.023040

Converted                                               1.000000
Tags_Will revert after reading the email                0.644307
Lead Profile_Potential Lead                             0.378061
Total Time Spent on Website                             0.362483
Last Notable Activity_SMS Sent                          0.351845
Lead Quality_Might be                                   0.342988
Last Activity_SMS Sent                                  0.325600
Lead Origin_Lead Add Form                               0.321702
What is your current occupation_Working Professional    0.313837
Lead Source_Reference                                   0.270830

Results: Data leakage identified in Tags_* Features, Last Activity, Last Notable Activity, and Lead Quality features.
These features are likely human post-conversion scores and therefore spoil the model's ability to generalize to new leads.
To address this, we will drop these features and retrain the model to ensure it learns from pre-conversion signals only.
--------------------- Data Leakage Analysis ---------------------


--------------------- Model Confidence ---------------------
Displaying the distribution of predicted probabilities to understand the model's confidence levels and to help in setting an appropriate threshold for lead scoring.
print(y_proba.min(), y_proba.max()) 
0.00013610424260684584 Minimum
0.9998376128306323 Maximum
This wide range indicates that the model is able to differentiate between leads with very low and very high conversion probabilities, which is a good sign.

Display the distribution of predicted probabilities:
import numpy as np
print(np.percentile(y_proba, [10, 50, 90]))
[0.00755532 0.12666112 0.99218659]

This shows that 10% of leads have a predicted probability below 0.0075, 50% are below 0.126, and 90% are below 0.992.
The model strongly differentiates between low- and high-quality leads, with the top 10% having near-certain conversion probabilities.
But, the high accuracy also raises concerns regarding overfitting, so we will compare results with a random forest model and a gradient boosting model.
--------------------- Model Confidence ---------------------


--------------------- Model Comparison ---------------------
Modell	            Accuracy	  Precision (1)	    Recall (1)	    Charakter
Logistic Regression	0.88	      0.98	            0.72	        very conservative
Random Forest	    0.92	      0.93	            0.88	        balanced, but slightly optimistic
Gradient Boosting	0.92	      0.96	            0.84	        stable & realistic

Score-Verteilung Vergleich
Modell	   Min	   Median	   90%	   Interpretation
LogReg	   ~0.0001	0.13	      0.99	   very selective
RF	       0.0	    0.19	      0.99	   optimistic
GB	       0.008 	0.13	      0.98	   stable & realistic

We evaluated multiple models and selected Gradient Boosting as it provides the best balance between precision and recall, 
making it suitable for prioritizing leads while still capturing a large portion of potential conversions.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Modell
rf = RandomForestClassifier(random_state=42)

# Trainieren
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred_rf))
y_proba_rf = rf.predict_proba(X_test)[:, 1]
print(y_proba_rf.min(), y_proba_rf.max())
print(np.percentile(y_proba_rf, [10, 50, 90]))

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1] # Get the predicted probabilities for the positive class (conversion)
y_pred_custom = (y_proba > 0.9).astype(int) # Classify as 1 (converted) if probability > 0.9, otherwise classify as 0 (not converted)

# Predict on the test set
#y_pred = model.predict(X_test)

# Evaluate the model using classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_custom))

lead_score = (y_proba * 100).round().astype(int)
print("Lead Scores (0-100):")
print(lead_score[:10])

# Cross-validation to evaluate model performance more robustly across different subsets of the data.
from sklearn.model_selection import cross_validate
scores = cross_validate(gb,X,y,cv=5,scoring=['accuracy', 'precision', 'recall', 'f1'])

print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision'].mean())
print("Recall:", scores['test_recall'].mean())
print("F1:", scores['test_f1'].mean())

# Evaluation
print(classification_report(y_test, y_pred_gb))
y_proba_gb = gb.predict_proba(X_test)[:, 1]
print(y_proba_gb.min(), y_proba_gb.max())
print(np.percentile(y_proba_gb, [10, 50, 90]))

Cross Valdation Results:
Accuracy: 0.8244588744588744
Precision: 0.7816541698160097
Recall: 0.7601710607182818
F1: 0.7695831670388026
--------------------- Model Comparison ---------------------

--------------------- Model Interpretability ---------------------
# SHAP values for model interpretability.
import shap

explainer = shap.TreeExplainer(gb)
shap_values = explainer.shap_values(X_test)

print("SHAP Summary Plot:")
print(shap.summary_plot(shap_values, X_test))
--------------------- Model Interpretability ---------------------


--------------------- Threshold Optimization Results ---------------------

Threshold | Precision | Recall | F1-Score | Interpretation
----------------------------------------------------------------------
0.1       | 0.549     | 0.974  | 0.702    | Very aggressive, many false positives
0.2       | 0.691     | 0.910  | 0.785    | High recall, useful for growth-focused strategy
0.3       | 0.740     | 0.853  | 0.792    | Strong balance, slightly recall-focused
0.4       | 0.784     | 0.815  | 0.799    | Best overall balance (recommended)
0.5       | 0.806     | 0.781  | 0.794    | Default threshold, slightly precision-focused
0.6       | 0.855     | 0.692  | 0.765    | More conservative lead selection
0.7       | 0.934     | 0.553  | 0.695    | High-confidence leads only
0.8       | 0.957     | 0.482  | 0.641    | Extremely selective
0.9       | 0.971     | 0.320  | 0.481    | Almost only perfect leads selected

Conclusion:
- Threshold 0.4 achieved the best F1-score and provided the best balance
  between Precision and Recall for a general business use case.
- Higher thresholds increase Precision but reduce Recall.
- Lower thresholds increase Recall but generate more false positives.

# Threshold Optimization
from sklearn.metrics import precision_score, recall_score, f1_score
thresholds = np.arange(0.1, 1.0, 0.1)
print("Thresholds:")
for t in thresholds:

    y_pred_custom = (y_proba_gb >= t).astype(int)

    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)

    print(f"Threshold: {t:.1f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print("-----")
--------------------- Threshold Optimization Results ---------------------


--------------------- Calibration and ROC Analysis ---------------------
# Calibration curve to evaluate how well the predicted probabilities align with actual outcomes.
prob_true, prob_pred = calibration_curve(y_test,y_proba_gb,n_bins=10)

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")

plt.title("Calibration Plot")
plt.show()

# Recall-Precision Curve 
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test,y_proba_gb)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Calibration curve to evaluate how well the predicted probabilities align with actual outcomes.
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_proba_gb)
auc_score = roc_auc_score(y_test, y_proba_gb)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
--------------------- Calibration and ROC Analysis ---------------------


--------------------- Hyperparameter Tuning ---------------------
# Grid Search for Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'n_estimators': [50, 100],'learning_rate': [0.05, 0.1],'max_depth': [2, 3]}
gb = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(gb,param_grid,cv=5,scoring='f1',n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest F1 Score:")
print(grid_search.best_score_)
--------------------- Hyperparameter Tuning ---------------------


--------------------- Feature Selection ---------------------
# Feature importance analysis to identify which features are most influential in the model's predictions.
feature_importance = pd.Series(final_model.feature_importances_,index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
print(feature_importance.head(20))

Total Time Spent on Website                             0.329089
Lead Origin_Lead Add Form                               0.224584
Lead Profile_Potential Lead                             0.122259
What is your current occupation_Working Professional    0.071668
Asymmetrique Activity Score_13.0                        0.033282
Lead Profile_Select                                     0.030218
Asymmetrique Activity Score_15.0                        0.030055
TotalVisits                                             0.021792
Do Not Email_Yes                                        0.017706
Asymmetrique Activity Index_03.Low                      0.016385
Asymmetrique Activity Score_16.0                        0.013106
How did you hear about X Education_Select               0.011357
What is your current occupation_Unemployed              0.010596
Page Views Per Visit                                    0.010140
Lead Profile_Student of SomeSchool                      0.008510
Specialization_Select                                   0.007832
Asymmetrique Profile Score_Unknown                      0.004707
Asymmetrique Activity Score_Unknown                     0.004062
City_Select                                             0.004028
Asymmetrique Activity Score_17.0                        0.003987

Decided to remove features with importance below 0.005 to simplify the model and reduce noise, while retaining the most influential features for lead scoring.
After removing, the model's performance did not significantly decrease, indicating that the removed features were not contributing much to the predictive power of the model.

              precision    recall  f1-score   support

           0       0.87      0.84      0.86      1107
           1       0.78      0.82      0.80       741

    accuracy                           0.83      1848
   macro avg       0.83      0.83      0.83      1848
weighted avg       0.84      0.83      0.83      1848
--------------------- Feature Selection ---------------------


--------------------- RFE (Recursive Feature Elimination) ---------------------
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier

# Base model
gb_rfe = GradientBoostingClassifier(random_state=42)

# RFE
rfe = RFE(estimator=gb_rfe,n_features_to_select=20)

# Fit RFE
rfe.fit(X, y)

# Selected features
selected_features = X.columns[rfe.support_]

print("Selected Features:")
print(selected_features)

X_rfe = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_rfe,y,test_size=0.2,random_state=42)
gb_final = GradientBoostingClassifier(random_state=42)
gb_final.fit(X_train, y_train)

y_proba = gb_final.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.4).astype(int)

print(classification_report(y_test, y_pred))

Index(['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
       'Lead Origin_Landing Page Submission', 'Lead Origin_Lead Add Form',
       'Do Not Email_Yes', 'Specialization_Select',
       'How did you hear about X Education_Select',
       'What is your current occupation_Unemployed',
       'What is your current occupation_Working Professional',
       'Lead Profile_Potential Lead', 'Lead Profile_Select',
       'Lead Profile_Student of SomeSchool', 'City_Select',
       'Asymmetrique Activity Index_03.Low',
       'Asymmetrique Activity Score_13.0', 'Asymmetrique Activity Score_15.0',
       'Asymmetrique Activity Score_16.0', 'Asymmetrique Activity Score_17.0',
       'Asymmetrique Profile Score_Unknown'],
      dtype='str')
              precision    recall  f1-score   support

           0       0.87      0.85      0.86      1107
           1       0.79      0.82      0.80       741

    accuracy                           0.84      1848
   macro avg       0.83      0.83      0.83      1848
weighted avg       0.84      0.84      0.84      1848

RFE was able to identify a subset of 20 features that maintained strong model performance while reducing complexity.
Therefore we will train the final model using these selected features to ensure a more interpretable and efficient lead scoring model.
--------------------- RFE (Recursive Feature Elimination) ---------------------


--------------------- XGBoost Result ---------------------
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

xgb = XGBClassifier(random_state=42,eval_metric='logloss')
xgb.fit(X_train_r, y_train_r)

y_proba_xgb = xgb.predict_proba(X_test_r)[:, 1]
threshold = 0.4
y_pred_xgb = (y_proba_xgb >= threshold).astype(int)

print(classification_report(y_test, y_pred_xgb))
--------------------- XGBoost Result ---------------------

'''