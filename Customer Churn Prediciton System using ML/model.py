import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv(r"C:\Users\K anusree\OneDrive\Desktop\FSDS\PROJECTS\CAPSTONEPROJECTS\Telecom Customer Churn\Churn.csv")

# convert categorical columns
df['Intl_Plan'] = df['Intl_Plan'].map({'yes':1,'no':0})
df['Vmail_Plan'] = df['Vmail_Plan'].map({'yes':1,'no':0})
df['Churn'] = df['Churn'].map({'yes':1,'no':0})

# features
X = df[['Day_Mins','Eve_Mins','Night_Mins','Intl_Mins','CustServ_Calls','Intl_Plan','Vmail_Plan']]
y = df['Churn']

# split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# save model
pickle.dump(model,open("Churn_model.pkl","wb"))

print("Model trained successfully")