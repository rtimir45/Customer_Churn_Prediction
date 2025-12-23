# Customer_Churn_Prediction
Customer Churn Prediction using python


---

# Customer Churn Prediction Using Machine Learning

# Project Overview

Customer churn refers to customers who stop using a company’s service.
This project builds a **machine learning classification model** to predict whether a customer will **churn or not**, based on their demographic information, account details, and service usage.

Predicting churn helps businesses take proactive steps to retain customers.



# Objective

* Analyze customer data to understand churn patterns
* Build a machine learning model to predict customer churn
* Evaluate model performance using classification metrics


# Skills & Concepts Covered

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Encoding Categorical Variables
* Train-Test Split
* Logistic Regression (Classification)
* Model Evaluation


# Tools & Libraries Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**



# Dataset

* **Name:** Telco Customer Churn Dataset
* **File:** `customer_churn.csv`
* **Source:** Kaggle (public dataset)

# Key Features

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`
* `tenure`
* `Contract`
* `MonthlyCharges`
* `TotalCharges`

# Target Variable

  -`Churn` → Yes / No



## ⚙️ Data Preprocessing

* Converted `TotalCharges` to numeric
* Handled missing values
* Encoded categorical variables using `LabelEncoder`
* Removed unnecessary columns like `customerID`


# Machine Learning Model

 - **Algorithm Used:** Logistic Regression
 - **Problem Type:** Binary Classification



# Model Evaluation Metrics

 -Accuracy Score
 -Confusion Matrix
 -Precision, Recall, F1-Score

These metrics help evaluate how well the model predicts customer churn.



#Sample Prediction

The trained model can predict whether a customer is likely to churn based on their input features.

Example output:

```
Churn Prediction: Yes
```


# How to Run the Project

1. Install required libraries:

```bash
pip install -r requirements.txt
```

2. Run the Code file


# Results & Insights

 -Customers with **month-to-month contracts** show higher churn
 -Long-term customers are less likely to churn
 -Contract type and tenure are strong churn indicators



#Future Improvements

  -Use advanced models like Random Forest or XGBoost
  -Perform feature importance analysis



 #Tools: Python, Pandas, Seaborn, Scikit-learn



 #License

 This project is open-source and intended for educational purposes.
