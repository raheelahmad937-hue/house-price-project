# app.py - A tiny web app that uses our trained model to predict prices

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load our saved "brain" (the trained model)
model = joblib.load("house_price_model.pkl")

# Load the column structure so new data matches the training format
data = pd.read_csv("House_Price_Prediction_Dataset.csv")
data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"])
model_columns = data.drop(columns=["Id", "Price"]).columns

app = Flask(__name__)

@app.route("/")
def home():
    return "House Price Prediction API is running! Go to /predict to use it."

@app.route("/predict", methods=["POST"])
def predict():
    # Get the house info sent to us
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])

    # Turn words into numbers, same way as training
    input_df = pd.get_dummies(input_df)

    # Make sure it has the SAME columns as training data (fill missing with 0)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)


    # Predict!
    prediction = model.predict(input_df)[0]

    return jsonify({"predicted_price": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)