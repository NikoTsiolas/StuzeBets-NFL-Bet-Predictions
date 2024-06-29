from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model/stuzebets_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/moneyline')
def moneyline():
    return render_template('MoneyLine.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Prediction logic here
    # Example:
    # features = [request.form['feature1'], request.form['feature2'], ...]
    # prediction = model.predict([features])
    # return render_template('result.html', prediction=prediction)
    pass

if __name__ == '__main__':
    app.run(debug=True)
