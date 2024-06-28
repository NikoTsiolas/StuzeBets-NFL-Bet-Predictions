from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load your model
model = joblib.load('model/stuzebets_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    # Convert form data to DataFrame
    df = pd.DataFrame([data])
    # Make prediction
    prediction = model.predict(df)
    # Convert prediction to a list and return as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
