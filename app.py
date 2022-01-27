from flask import Flask,render_template,url_for,request
import pickle
import numpy as np

app=Flask(__name__)

model_path = 'Trained_Model/log_reg.pkl'
model = pickle.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    gender = int(request.form['gender'])
    status = int(request.form['married'])
    education = int(request.form['education'])
    work = int(request.form['employment'])
    app_income = int(request.form['applicant_income'])
    co_app_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_amount_term = float(request.form['loan_amount_term'])
    credit_history = float(request.form['credit_history'])
    property_area = int(request.form['property_area'])
    dependents = int(request.form['dependents'])

    query = np.array([[gender, status, education, work, app_income, 
                        co_app_income, loan_amount, loan_amount_term, credit_history, property_area, dependents]])

    prediction = model.predict(query)

    
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
