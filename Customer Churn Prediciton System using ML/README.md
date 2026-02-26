Customer Churn Prediction using Machine Learning
--
 Overview
 --
Customer churn is a major challenge in the telecom industry. This project builds a Machine Learning classification model to predict whether a customer will churn (leave the service) based on their usage patterns and service interactions.
Early churn prediction helps companies improve customer retention and reduce revenue loss.

---

 Problem Statement
--
Predict whether a telecom customer will churn using historical usage and service data.

Target Variable:
- churn (True / False)

---

 Dataset Description
--
The dataset contains customer usage behavior and service-related information.

 Key Features
 --

- vmail_days / number_vmail_messages – Voicemail usage
- eve_mins – Evening call minutes
- night_mins – Night call minutes
- intl_mins – International call minutes
- custserv – Number of customer service calls
- day_mins – Day call minutes
- total_charge – Billing information

---

 Technologies Used:
 --

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

Project Workflow :
--

1. Data Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature Scaling  
4. Train-Test Split  
5. Model Training  
6. Model Evaluation  

---

 Models Implemented :
--
- Logistic Regression
- Decision Tree
- Random Forest

---

 Model Performance :
-- 
| Model               | Accuracy | ROC-AUC |
|--------------------|----------|----------|
| Logistic Regression | XX%      | XX       |
| Random Forest       | XX%      | XX       |

(Update values with your actual results)

---

Evaluation Metrics :
--
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

---

How to Run the Project
--
 Install Required Libraries  
pip install pandas numpy scikit-learn matplotlib seaborn  
 Run the Script:  
 
python train.py

--
 Key Insights
--
- Customers with high intl_mins are more likely to churn.
- Higher custserv calls strongly indicate churn probability.
- Billing and contract type significantly influence churn behavior.

---

 Business Impact

- Identify high-risk customers early
- Improve retention strategy
- Optimize marketing campaigns
- Reduce customer acquisition costs

---

 Future Improvements

- Hyperparameter tuning
- XGBoost implementation
- Model deployment using Flask / Streamlit
- Real-time churn prediction API

---

 Domain

Machine Learning | Classification | Telecom Analytics
