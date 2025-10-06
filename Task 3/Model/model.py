from flask import Flask, render_template, request
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model/weather_model.pkl")
scaler = joblib.load("model/scaler.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        meanpressure = float(request.form['meanpressure'])

        # Prepare input for model
        input_data = np.array([[humidity, wind_speed, meanpressure]])
        input_scaled = scaler.transform(input_data)

        # Predict temperature
        prediction = model.predict(input_scaled)[0]
        output = round(prediction, 2)

        return render_template('index.html', prediction_text=f'Predicted Mean Temperature: {output}°C')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
