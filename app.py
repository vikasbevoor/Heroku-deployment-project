import pickle
import numpy as np
from flask import Flask, request, render_template,url_for
import xgboost as xgb

app = Flask(__name__)

terms = pickle.load(open('terms.obj','rb'))
grade = pickle.load(open('grade.obj','rb'))
home_ownership = pickle.load(open('home_ownership.obj','rb'))
verification_status = pickle.load(open('verification_status.obj','rb'))
purpose = pickle.load(open('purpose.obj','rb'))
initial_list_status = pickle.load(open('initial_list_status.obj','rb'))
Experience = pickle.load(open('Experience.obj','rb'))
model = pickle.load(open('finalized_model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])  
def predict():
    features=[]
    a = (float(request.form["loan_amnt"]))
    features.append(a)
    b=request.form["terms"]
    features.append(terms['terms'][b])
    c=(float(request.form["Rate_of_intrst"]))
    features.append(c)
    d=request.form["grade"]
    features.append(grade['grade'][d])
    e=request.form["home_ownership"]
    features.append(home_ownership['home_ownership'][e])
    f=(float(request.form["annual_inc"]))
    features.append(f)
    g=request.form["verification_status"]
    features.append(verification_status['verification_status'][g])
    h=request.form["purpose"]
    features.append(purpose['purpose'][h])
    i=(float(request.form["debt_income_ratio"]))
    features.append(i)
    j=(float(request.form["delinq_2yrs"]))
    features.append(j)
    k=(float(request.form["inq_last_6mths"]))
    features.append(k)
    l=(float(request.form["numb_credit"]))
    features.append(l)
    m=(float(request.form["pub_rec"]))
    features.append(m)
    n=(float(request.form["total_credits"]))
    features.append(n)
    o=request.form["initial_list_status"]
    features.append(initial_list_status['initial_list_status'][o])
    p=(float(request.form["total_rec_int"]))
    features.append(p)
    q=(float(request.form["total_rec_late_fee"]))
    features.append(q)
    r=(float(request.form["recoveries"]))
    features.append(r)
    s=request.form["Experience"]
    features.append(Experience['Experience'][s])
    t=(float(request.form["mths_since_last_delinq"]))
    features.append(t)
    u=(float(request.form["tot_curr_bal"]))
    features.append(u)
    v=(float(request.form["tot_colle_amt"]))
    features.append(v)

    final_features = np.array(features).reshape(-1,22)
    prediction = model.predict(final_features)
    return render_template('results.html',prediction=prediction)

# @app.route('/predict_csv',methods=['POST'])  
# def predict_csv():
#     features = []
#     final_features = np.array(features).reshape(-1,22)
#     prediction = model.predict(final_features)
#     return render_template('results.html',prediction=prediction)

@app.route('/manual_page',methods=['POST'])     
def manual_page():
    return render_template('manual_page.html')

@app.route('/read_file',methods=['POST']) 
def read_file():
    return render_template('read_file.html')

if __name__ == "__main__":
    app.run(debug=True)