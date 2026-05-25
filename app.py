import streamlit as st
import pandas as pd
from model_utils import predict_lead_score


st.set_page_config(
    page_title="Lead Scoring Dashboard",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Lead Scoring Dashboard")
st.markdown(
    """
    Use this tool to estimate how likely a lead is to convert and prioritize sales follow-up.
    """
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Lead Information")

    total_time = st.number_input("Total Time Spent on Website", min_value=0, value=800)
    total_visits = st.number_input("Total Visits", min_value=0, value=6)
    page_views = st.number_input("Page Views Per Visit", min_value=0.0, value=3.5)

    lead_origin = st.selectbox(
        "Lead Origin",
        ["Lead Add Form", "Landing Page Submission", "Lead Import", "Quick Add Form"]
    )

    do_not_email = st.selectbox("Do Not Email", ["No", "Yes"])

    occupation = st.selectbox(
        "Occupation",
        ["Working Professional", "Unemployed", "Student", "Other"]
    )

    lead_profile = st.selectbox(
        "Lead Profile",
        ["Potential Lead", "Select", "Student of SomeSchool", "Other"]
    )

    asym_activity_index = st.selectbox(
        "Asymmetrique Activity Index",
        ["03.Low", "02.Medium", "01.High", "Unknown"]
    )

    asym_activity_score = st.selectbox(
        "Asymmetrique Activity Score",
        [13.0, 15.0, 16.0, 17.0, "Unknown"]
    )

    asym_profile_score = st.selectbox(
        "Asymmetrique Profile Score",
        ["Unknown", 13.0, 15.0, 16.0, 17.0]
    )

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05
    )

    calculate = st.button("Calculate Lead Score")

    st.divider()

st.header("📁 Batch Lead Scoring")
st.write("Upload a CSV file to score multiple leads at once.")

uploaded_file = st.file_uploader(
    "Upload Lead CSV",
    type=["csv"]
)

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(batch_df.head())

    results = []

    for _, row in batch_df.iterrows():
        input_data = row.to_dict()

        score, _, category, explanations = predict_lead_score(input_data)

        probability = score / 100
        prediction = int(probability >= threshold)

        results.append({
            "Lead Score": score,
            "Prediction": "Likely to Convert" if prediction == 1 else "Unlikely to Convert",
            "Category": category
        })

    results_df = pd.DataFrame(results)

    output_df = pd.concat(
        [batch_df.reset_index(drop=True), results_df],
        axis=1
    )

    st.subheader("Scored Leads")
    st.dataframe(output_df)

    csv = output_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Scored Leads",
        data=csv,
        file_name="scored_leads.csv",
        mime="text/csv"
    )

with right_col:
    st.header("Prediction Result")

    if calculate:
        input_data = {
            "Total Time Spent on Website": total_time,
            "TotalVisits": total_visits,
            "Page Views Per Visit": page_views,
            "TotalVisits_missing": 0,

            "Lead Origin": lead_origin,
            "Do Not Email": do_not_email,
            "Specialization": "Select",
            "How did you hear about X Education": "Select",
            "What is your current occupation": occupation,
            "Lead Profile": lead_profile,
            "City": "Select",
            "Asymmetrique Activity Index": asym_activity_index,
            "Asymmetrique Activity Score": asym_activity_score,
            "Asymmetrique Profile Score": asym_profile_score
        }

        score, _, category, explanations = predict_lead_score(input_data)
        print(input_data)
        print("Score:", score)
        print("Category:", category)
        print("-----")
        probability = score / 100
        prediction = int(probability >= threshold)

        st.metric("Lead Score", f"{score}/100")
        st.progress(score / 100)

        if category == "Hot Lead":
            st.success("🔥 Hot Lead — High priority for Sales")
        elif category == "Warm Lead":
            st.warning("🌤️ Warm Lead — Worth nurturing or following up")
        else:
            st.error("❄️ Cold Lead — Low priority")

        if prediction == 1:
            st.success("✅ Prediction: Likely to Convert")
        else:
            st.error("❌ Prediction: Unlikely to Convert")

        st.caption(
            f"Current decision threshold: {threshold:.2f}. "
            "Lower threshold selects more leads; higher threshold selects only high-confidence leads."
        )

        st.subheader("Why this score?")

        for feature, value in explanations.items():
            if value > 0:
                st.write(f"🟢 {feature}: increased the conversion score")
            else:
                st.write(f"🔴 {feature}: decreased the conversion score")
    else:
        st.info("Enter lead information and click the button to calculate a score.")
    
    st.subheader("Feedback Loop")

    actual_outcome = st.selectbox(
        "Did this lead actually convert?",
        ["Not known yet", "Yes", "No"]
    )

    if st.button("Save Feedback"):
        if actual_outcome == "Not known yet":
            st.warning("Please select Yes or No before saving feedback.")
        else:
            feedback_row = input_data.copy()
            feedback_row["Lead Score"] = score
            feedback_row["Prediction"] = prediction
            feedback_row["Category"] = category
            feedback_row["Actual Converted"] = 1 if actual_outcome == "Yes" else 0

            feedback_df = pd.DataFrame([feedback_row])

            feedback_file = "feedback_data.csv"

            try:
                existing_feedback = pd.read_csv(feedback_file)
                feedback_df = pd.concat(
                    [existing_feedback, feedback_df],
                    ignore_index=True
            )
            except FileNotFoundError:
                pass

            feedback_df.to_csv(feedback_file, index=False)

            st.success("Feedback saved successfully!")
