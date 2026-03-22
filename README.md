# 💼 Job Salary Prediction Using Machine Learning

## 📌 Project Overview
This project is an End-to-End Machine Learning Salary Prediction System that predicts job salaries based on job-related and candidate-related features such as job title, experience, education level, skills, industry, company size, location, remote work, and certifications.

The project includes:
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Outlier Detection & Treatment
- Feature Encoding
- Feature Scaling
- Training Multiple Regression Models
- Model Comparison
- Best Model Selection
- Model Saving
- Streamlit Web Application Deployment

---

## 🧠 Problem Statement
The objective of this project is to build a machine learning model that can accurately predict the salary of a job based on various job and candidate attributes.

---

## 📊 Dataset Features

| Feature | Description |
|--------|-------------|
| job_title | Job Role |
| experience_years | Years of Experience |
| education_level | Education Qualification |
| skills_count | Number of Skills |
| industry | Industry Type |
| company_size | Company Size |
| location | Job Location |
| remote_work | Remote or Onsite |
| certifications | Number of Certifications |
| salary | Target Variable |

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was performed to understand:
- Salary distribution
- Relationship between experience and salary
- Salary by job title, education, industry, company size
- Outliers in numerical features
- Correlation between numerical features

Outliers were detected using the IQR method and treated using capping.

---

## ⚙️ Machine Learning Algorithms Used
Multiple regression algorithms were trained and compared:

1. Linear Regression  
2. Ridge Regression  
3. Lasso Regression  
4. KNN Regressor  
5. Support Vector Regressor (SVR)  
6. Decision Tree Regressor  
7. Random Forest Regressor  
8. Gradient Boosting Regressor  
9. XGBoost Regressor  

Models were evaluated using:
- R2 Score
- RMSE
- MAE

The best-performing model was selected based on highest R2 Score and lowest RMSE.

---

## 🏆 Model Evaluation Metrics

| Metric | Description |
|-------|-------------|
| R2 Score | Model accuracy |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |

---

## 🧱 Machine Learning Pipeline
The pipeline includes:
- OneHotEncoder for categorical features
- StandardScaler for numerical features
- Regression model

This prevents data leakage and ensures consistent preprocessing.

---

## 💻 Streamlit Web Application
A Streamlit web app was created where users can:
- Select job title
- Enter experience
- Enter skills
- Enter certifications
- Select education, industry, company size, location, remote work
- Get predicted salary

---

## 📁 Project Structure
Salary_Prediction_Project/
│
├── app.py
├── job_salary_prediction_dataset.csv
├── SalaryPrediction.pkl_XGBRegressor
├── ML_FinalTask_on_Salary_Prediction.ipynb
├── README.md
├── requirements.txt


---

## 📈 Machine Learning Workflow
1. Import Libraries  
2. Load Dataset  
3. Data Cleaning  
4. Exploratory Data Analysis  
5. Outlier Detection & Treatment  
6. Feature Encoding  
7. Feature Scaling  
8. Train-Test Split  
9. Model Training  
10. Model Evaluation  
11. Model Comparison  
12. Best Model Selection  
13. Save Model  
14. Deploy using Streamlit  

---

## 🎯 Conclusion
Multiple regression algorithms were trained and evaluated. Ensemble models such as Random Forest, Gradient Boosting, and XGBoost performed better than linear models. The best model was selected based on evaluation metrics and deployed using Streamlit for real-time salary prediction.

---

## 🚀 Future Improvements
- Hyperparameter tuning
- Cross-validation
- Feature engineering
- Model deployment on cloud
- Add salary insights dashboard

---

## 📦 Requirements
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- joblib
- seaborn
- matplotlib

---

## 👨‍💻 Author
Machine Learning Salary Prediction Project  
End-to-End ML Project with Streamlit Deployment
