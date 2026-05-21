from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso

# Si lo pongo al principio me coloco en el directorio en el que está mi script. Así consigo que 
# se ejecute desde ahi y si luego accedo a cosas (modelo, ficheros...)
os.chdir (os.path.dirname (__file__))

app = Flask(__name__)
app.config["DEBUG"] = True # en el momento en el que tenga la api funcionando esto se pone a False. Que
                           # deje de debuggear 

# Ligado al endopoint "/" o sea el home, con el método GET
@app.route('/', methods=['GET'])
def hello(): 
    return "Bienvenido a mi API del modelo advertising"

# Ligado al endpoint '/api/v1/predict', con el método GET
@app.route ('/api/v1/predict', methods=['GET'])
def predict(): 

    model = pickle.load(open('ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
    
    return jsonify({'predictions': prediction[0]})

# Rutarlo al endpoint '/api/v1/retrain/', metodo GET
@app.route ('/api/v1/retrain', methods=['GET'])
def retrain(): 
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        pickle.dump(model, open('ad_model.pkl', 'wb'))

        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

# Faltaría un endpoint nuevo en el que pudieramos subir una fila al csv y reentrenar el modelo
app.run()