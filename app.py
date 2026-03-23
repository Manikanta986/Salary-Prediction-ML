import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Salary Prediction App")

st.title("Salary Prediction App")

# Load dataset
df = pd.read_csv("job_salary_prediction_dataset.csv")

# Load model
model = joblib.load("model.pkl")

st.sidebar.header("Enter Details")

job_title = st.sidebar.selectbox("Job Title", df["job_title"].unique())
education = st.sidebar.selectbox("Education Level", df["education_level"].unique())
industry = st.sidebar.selectbox("Industry", df["industry"].unique())
company_size = st.sidebar.selectbox("Company Size", df["company_size"].unique())
location = st.sidebar.selectbox("Location", df["location"].unique())
remote = st.sidebar.selectbox("Remote Work", df["remote_work"].unique())

experience = st.sidebar.slider("Experience Years", 0, 30, 1)
skills = st.sidebar.slider("Skills Count", 0, 20, 1)
certifications = st.sidebar.slider("Certifications", 0, 10, 0)

if st.sidebar.button("Predict Salary"):

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

    st.subheader("Predicted Salary")
    st.success(f"₹ {salary:,}")
