from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

data = pd.read_csv("House_Price_Prediction_Dataset.csv")
data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"])
model_columns = data.drop(columns=["Id", "Price"]).columns

app = Flask(__name__)

# Simple webpage with a form
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body { font-family: Arial; max-width: 500px; margin: 50px auto; padding: 20px; }
        input, select { width: 100%; padding: 8px; margin: 6px 0 14px 0; box-sizing: border-box; }
        button { background: #2c7be5; color: white; padding: 10px; border: none; width: 100%; cursor: pointer; }
        #result { margin-top: 20px; font-size: 1.3em; font-weight: bold; }
    </style>
</head>
<body>
    <h2>🏠 House Price Predictor</h2>
    <form id="predictForm">
        Area (sq ft): <input type="number" id="Area" value="2000" required>
        Bedrooms: <input type="number" id="Bedrooms" value="3" required>
        Bathrooms: <input type="number" id="Bathrooms" value="2" required>
        Floors: <input type="number" id="Floors" value="1" required>

        Year Built: <input type="number" id="YearBuilt" value="2005" required>
        Location:
        <select id="Location">
            <option>Downtown</option>
            <option>Suburban</option>
            <option>Rural</option>
        </select>
        Condition:
        <select id="Condition">
            <option>Excellent</option>
            <option>Good</option>
            <option>Fair</option>
        </select>
        Garage:
        <select id="Garage">
            <option>Yes</option>
            <option>No</option>
        </select>
        <button type="submit">Predict Price</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById("predictForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            const payload = {
                Area: Number(document.getElementById("Area").value),
                Bedrooms: Number(document.getElementById("Bedrooms").value),
                Bathrooms: Number(document.getElementById("Bathrooms").value),
                Floors: Number(document.getElementById("Floors").value),
                YearBuilt: Number(document.getElementById("YearBuilt").value),
                Location: document.getElementById("Location").value,

                Condition: document.getElementById("Condition").value,
                Garage: document.getElementById("Garage").value
            };
            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            document.getElementById("result").innerText = "💰 Predicted Price: $" + data.predicted_price.toLocaleString();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_df)[0]
    return jsonify({"predicted_price": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)