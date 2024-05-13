from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import pickle

app = Flask(__name__)
with open('random_forest.pkl','rb') as f:
    model=pickle.load(f)


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/form')
def show_form():
    return render_template('form.html')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            values = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
            prediction = model.predict(values)
            print(prediction)    
            return render_template('result.html', prediction=prediction)
    except:
        return render_template('index.html')

@app.route('/visualizations')
def visualizations():
    # Render visualization page
    return render_template('visualization.html')

if __name__ == '__main__':
    app.run(debug=True)
