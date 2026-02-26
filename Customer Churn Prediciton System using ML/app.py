from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# load trained model
model = pickle.load(open(r"C:\Users\K anusree\OneDrive\Desktop\FSDS\PROJECTS\CAPSTONEPROJECTS\Telecom Customer Churn\Churn_model.pkl", "rb"))


# home page
@app.route('/')
def home():
    return render_template("index.html")


# prediction route
@app.route('/predict', methods=['POST'])
def predict():

    Day_Mins = float(request.form['Day_Mins'])
    Eve_Mins = float(request.form['Eve_Mins'])
    Night_Mins = float(request.form['Night_Mins'])
    Intl_Mins = float(request.form['Intl_Mins'])
    CustServ_Calls = float(request.form['CustServ_Calls'])
    Intl_Plan = float(request.form['Intl_Plan'])
    Vmail_Plan = float(request.form['Vmail_Plan'])

    # order must match training features
    features = [[
        Day_Mins,
        Eve_Mins,
        Night_Mins,
        Intl_Mins,
        CustServ_Calls,
        Intl_Plan,
        Vmail_Plan
    ]]

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Customer Will Churn"
    else:
        result = "Customer Will Stay"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
        
        
    