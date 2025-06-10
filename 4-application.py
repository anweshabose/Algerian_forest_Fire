# run the code: python 4-application.py
import pickle
from flask import Flask,request,jsonify,render_template # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Initializing Flask
application=Flask(__name__)
app = application

## import ridge regressor model and standard scaler pickle
ridge_model=pickle.load(open('ridge.pkl','rb'))        # pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('scaler.pkl','rb'))     # pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html') # it will try to find "index.html" file inside the "templates" folder. So make the "index.html" file inside "templates" folder beforehand.

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        # pass

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0") # "0.0.0.0" is the host address # it is getting mapped with the local ip address of any machine that you are working # Obviously, local ip address is not publicly available.


# http://192.168.29.201:5000  this entire code is running in system which is having this ip adress.

# http://192.168.29.201:5000/predictdata when we are searching for predictdata in google chrome, it is a GET request. So home.html file will open in google chrome
# when we hit the predict button, it is a POST request and then prediction will take place


# Deployment in github repository

# Deployment in AWS:
# make python.config file inside .ebextensions folder
# bcz whenever we are in linuk, we need a python environment

# WSGIPath: 4-application:application => 4-application reffering to 4-application.py file & application is reffering to application=Flask(__name__) of this file