from model_utils import predict_lead_score

test_lead = {
    "Total Time Spent on Website": 3000,
    "TotalVisits": 15,
    "Page Views Per Visit": 8,
    "TotalVisits_missing": 0,

    "Lead Origin": "Lead Add Form",
    "Do Not Email": "No",
    "Specialization": "Select",
    "How did you hear about X Education": "Select",
    "What is your current occupation": "Working Professional",
    "Lead Profile": "Potential Lead",
    "City": "Select",
    "Asymmetrique Activity Index": "01.High",
    "Asymmetrique Activity Score": 15.0,
    "Asymmetrique Profile Score": "Unknown"
}


score, prediction, category, explanations = predict_lead_score(test_lead)

print("Lead Score:", score)
print("Prediction:", prediction)
print("Category:", category)

print("\nTop Explanations:")
print(explanations)