
# 📊 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to leave (churn) based on their demographic and banking behavior data. This helps businesses take proactive steps to retain customers.

---

## 🚀 Project Overview

Customer churn is a critical problem in industries like banking and telecom. In this project, we:

* Analyze customer data
* Perform data preprocessing and cleaning
* Train a Machine Learning model
* Predict whether a customer will leave or stay

---

## 📂 Dataset

* File: `Customer_Churn.csv`
* Contains customer-related attributes such as:

  * Credit Score
  * Geography
  * Gender
  * Age
  * Tenure
  * Balance
  * Number of Products
  * Credit Card Status
  * Active Membership
  * Estimated Salary
  * Target Variable: `Leave` (Churn)

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas
* Scikit-learn
* XGBoost
* NumPy

---

## 🧠 Machine Learning Model

* **Algorithm Used:** Random Forest Classifier
* **Alternative Model :** XGBoost Classifier
* **Train-Test Split:** 80% training / 20% testing
* **Evaluation Metrics:**

  * Accuracy Score
  * Classification Report
  * Confusion Matrix

---

## 🔄 Workflow

1. **Data Loading**

   ```python
   pd.read_csv('Customer_Churn.csv')
   ```

2. **Data Exploration**

   * Head, info, description
   * Null value check
   * Duplicate check

3. **Data Cleaning**

   * Filled missing values:

     * Mean → Numerical columns
     * Mode → Categorical columns

4. **Feature Encoding**

   * Label Encoding for:

     * Gender
     * Geography

5. **Feature Selection**

   * Dropped unnecessary column: `Surname`

6. **Model Training**

   ```python
   RandomForestClassifier(n_estimators=500)
   ```

7. **Model Evaluation**

   * Accuracy Score
   * Classification Report
   * Confusion Matrix

---

## 📈 Results

* Model achieves 86% accuracy in predicting customer churn
* Provides insights into customer retention patterns

---

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:

   ```bash
   pip install pandas scikit-learn xgboost
   ```

3. Run the script:

   ```bash
   python ccp.py
   ```

---

## 📌 Future Improvements

* Hyperparameter tuning
* Feature engineering
* Model comparison (XGBoost vs Random Forest)
* Deployment using Streamlit
* Real-time prediction system

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is open-source and  for educational purposes.


