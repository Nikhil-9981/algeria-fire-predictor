import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
ridge_model = pickle.load(open("models/Ridge_regressor.pkl",'rb'))
standard_scaler = pickle.load(open("models/Ridge_scaler.pkl",'rb'))

temperature = float(5)
RH = float(4)
Ws = float(1)
Rain = float(11)
FFMC = float(30)
ISI = float(3)
Classes = float(1)
Region = float(1.5)
        
new_data_scaled = standard_scaler.transform([['temperature', 'RH', 'Ws','Rain','FFMC','DMC','ISI','Classes','Region']])
result = ridge_model.predict(new_data_scaled)
print(result)