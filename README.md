# Predicting-heart-disease-using-ML
## 🫀 Heart Disease Prediction using Machine Learning
This project uses machine learning techniques to predict the presence of heart disease in patients based on various health parameters. The goal is to apply and evaluate different classification algorithms to identify patterns and insights from the dataset and make accurate predictions.

## 📌 Project Workflow
The project follows a systematic data science process:

1).Problem Definition

Predict whether a patient has heart disease based on their medical attributes.

2).Data Collection

The dataset used is heart-disease.csv, consisting of 303 entries and 14 features including age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate, and more.

3).Exploratory Data Analysis (EDA)

Examined feature distributions and data types

Visualized heart disease frequency by gender, chest pain type, and max heart rate

Checked for missing values and outliers

Engineered relevant features to enhance model performance

4).Model Building

Trained and evaluated three different machine learning models:

1).Logistic Regression

2).K-Nearest Neighbors (KNN)

3).Random Forest Classifier

5).Model Evaluation & Tuning

Used confusion matrix, precision, recall, and F1-score to evaluate models

Applied cross-validation for more reliable results

Performed hyperparameter tuning using GridSearchCV and RandomizedSearchCV

Analyzed feature importance to interpret the model

## ✅ Results
Achieved strong classification performance using Random Forest and Logistic Regression

Cross-validation and hyperparameter tuning improved model accuracy and generalization

Visualization helped in understanding data patterns and model decisions

## 📁 Technologies Used
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (Logistic Regression, KNN, Random Forest, GridSearchCV, etc.)

## 🚀 How to Run
Clone the repository

Install required packages (pip install -r requirements.txt)

Run the notebook: Predicting-heart-disease-using-ML.ipynb in Jupyter Notebook
