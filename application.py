import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standrad_scaler=pickle.load(open('models/standrad_scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        temperature = float(request.form['temperature'])
        rh = float(request.form['rh'])
        ws = float(request.form['ws'])
        rain = float(request.form['rain'])
        ffmc = float(request.form['ffmc'])
        dmc= float(request.form['dmc'])
        isi = float(request.form['isi'])
        classes = int(request.form['classes'])
        region = int(request.form['region'])
        data = [[
            temperature, rh, ws, rain,
            ffmc, dmc, isi,
            classes, region
        ]]
        scaled = standrad_scaler.transform(data)
        prediction = ridge_model.predict(scaled)[0]

        return render_template('predict_output.html',prediction=prediction)

    else:
        return render_template('predict_input.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)