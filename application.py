from flask import Flask,jsonify,request,render_template
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
application = Flask(__name__)
app=application

ridge_model=pickle.load(open('C:\\ML1\\models\\ridge.pkl','rb'))
standard_scaler=pickle.load(open('C:\\ML1\\models\\scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=["GET", "POST"])
def predict_data():
     if request.method == "POST":
          # Get the input values from the form
          Temperature = float(request.form.get('Temperature'))
          Rain = float(request.form.get('Rain'))
          FFMC = float(request.form.get('FFMC'))
          DMC = float(request.form.get('DMC'))
          DC = float(request.form.get('DC'))
          ISI = float(request.form.get('ISI'))
          BUI = float(request.form.get('BUI'))
          Classes = float(request.form.get('Classes'))
          Region = float(request.form.get('Region'))

          # Create a feature array
          features = np.array([Temperature, Rain, FFMC, DMC, DC, ISI, BUI, Classes, Region]).reshape(1, -1)

          # Scale the features
          features_scaled = standard_scaler.transform(features)

          # Make the prediction
          prediction = ridge_model.predict(features_scaled)

          # Return the prediction result
          return render_template('home.html', prediction=prediction[0])
     else:
          return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
