import streamlit as st
import pandas as pd
import joblib
import xgboost
import numpy as np

# Page settings
st.set_page_config(page_title="Salary Prediction App", layout="centered")

# Title
st.title("💼 Job Salary Prediction System")
st.write("Predict salary based on job details using Machine Learning")

# Load dataset and model
df = pd.read_csv("job_salary_prediction_dataset.csv")
model = joblib.load("SalaryPrediction.pkl_XGBRegressor")

# Sidebar Inputs
st.sidebar.header("Enter Job Details")

job_title = st.sidebar.selectbox("Job Title", sorted(df["job_title"].unique()))
education = st.sidebar.selectbox("Education Level", sorted(df["education_level"].unique()))
industry = st.sidebar.selectbox("Industry", sorted(df["industry"].unique()))
company_size = st.sidebar.selectbox("Company Size", sorted(df["company_size"].unique()))
location = st.sidebar.selectbox("Location", sorted(df["location"].unique()))
remote = st.sidebar.selectbox("Remote Work", sorted(df["remote_work"].unique()))

experience = st.sidebar.slider("Experience Years", 0, 30, 1)
skills = st.sidebar.slider("Skills Count", 0, 20, 1)
certifications = st.sidebar.slider("Certifications", 0, 10, 0)

# Prediction Button
if st.sidebar.button("Predict Salary"):
    
    # Validation
    if experience == 0 and skills == 0 and certifications == 0:
        st.warning("⚠ Please enter at least some experience, skills, or certifications.")
    
    else:
        input_data = pd.DataFrame({
            'job_title': [job_title],
            'experience_years': [experience],
            'education_level': [education],
            'skills_count': [skills],
            'industry': [industry],
            'company_size': [company_size],
            'location': [location],
            'remote_work': [remote],
            'certifications': [certifications]
        })

        prediction = model.predict(input_data)
        salary = int(prediction[0])

        # Fresher salary adjustment
        if experience == 0:
            salary = int(salary * 0.65)

        st.subheader("💰 Predicted Salary")
        st.success(f"₹ {salary:,}")

# Footer
st.write("---")
st.write("Machine Learning Salary Prediction Project | Streamlit App")
