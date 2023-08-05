import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging


logging.basicConfig(filename='logging.log' ,level=logging.INFO, format= '%(asctime)s %(name)s %(levelname)s %(message)s')
 


from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://Nikhil-9981:nikhil@cluster0.r6x7pnj.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    logging.info("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
     logging.info(str(e))

FWI_Prediction1 = client['FWI_Prediction']
Predicted_values1 = FWI_Prediction1['Predicted_values']


application = Flask(__name__)

app = application

# import ridge regressor and standat scaler pickl
ridge_model = pickle.load(open("models/Ridge_regressor.pkl",'rb'))
standard_scaler = pickle.load(open("models/Ridge_scaler.pkl",'rb'))

# Route for home page
@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict_data', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            
            temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))
            


            new_data_scaled = standard_scaler.transform([[temperature, RH, Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
            result = ridge_model.predict(new_data_scaled)
            
            
            result1 =float(result[0])
            mydic = {'temperature' : temperature , 'RH': RH,'Ws' : Ws, 'Rain' : Rain,'FFMC' : FFMC,'DMC' : DMC,
                    'ISI' : ISI,'Classes' : Classes, 'Region' :Region  , 'The Predicted FWI value is :' : result1 }
            logging.info("my FWI result is:  {}".format(mydic))
             
            Predicted_values1.insert_one(mydic)
            return render_template('home.html',result = result[0])



             

        except Exception as e:
            logging.info("Error is: "+str(e))
            return render_template('error.html', error=str(e))
    else :
        return render_template('home.html')


if __name__=="__main__":
 
    app.run(debug = True , host="0.0.0.0",port=8000 )
