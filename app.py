# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model_loan.pkl', 'rb') as file:
    classifier = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    features = [float(x) for x in request.form.values()]
    # Reshape the data for prediction
    features_array = np.array(features).reshape(1, -1)
    # Predict
    prediction = classifier.predict(features_array)
    if prediction[0] == 0:
        result = "Loan not approved."
    else:
        result = "Loan approved."
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run()
