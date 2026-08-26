from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

model = joblib.load("house_price_model.pkl")

data = pd.read_csv("House_Price_Prediction_Dataset.csv")
data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"])
model_columns = data.drop(columns=["Id", "Price"]).columns

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>House Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b0d14;
    --bg-glow: #1a1428;
    --panel: rgba(255,255,255,0.05);
    --panel-border: rgba(255,190,120,0.18);
    --amber: #ffb454;
    --amber-soft: #ffd9a0;
    --text: #f2ede4;
    --text-dim: #9a93a8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    background: radial-gradient(circle at 50% 20%, var(--bg-glow), var(--bg) 60%);
    font-family: 'Inter', sans-serif;
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    overflow-x: hidden;
  }
  .glow {
    position: fixed;
    top: -10%;
    left: 50%;
    transform: translateX(-50%);
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(255,180,84,0.35) 0%, rgba(255,180,84,0.08) 40%, transparent 70%);
    filter: blur(20px);
    pointer-events: none;
    animation: pulse 6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 0.7; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.08); }
  }
  .card {
    position: relative;
    width: 100%;
    max-width: 460px;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 40px 36px;
    backdrop-filter: blur(18px);
    box-shadow: 0 0 60px rgba(255,180,84,0.08), 0 20px 60px rgba(0,0,0,0.5);
  }
  .eyebrow {
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 10px;
    font-weight: 600;
  }
  h1 {
    font-family: 'Fraunces', serif;
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 8px 0;
    line-height: 1.15;
    background: linear-gradient(180deg, #fff, var(--amber-soft));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .sub {
    color: var(--text-dim);
    font-size: 14px;
    margin-bottom: 28px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  .field { margin-bottom: 14px; }
  .field.full { grid-column: 1 / -1; }
  label {
    display: block;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
  }
  input, select {
    width: 100%;
    padding: 11px 12px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    color: var(--text);
    font-size: 14px;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  input:focus, select:focus {
    outline: none;
    border-color: var(--amber);
    box-shadow: 0 0 0 3px rgba(255,180,84,0.15);
  }
  select option { background: #1a1a24; color: var(--text); }
  button {
    width: 100%;
    margin-top: 10px;
    padding: 14px;
    border: none;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--amber), #e8873f);
    color: #1a0f00;
    font-weight: 600;
    font-size: 15px;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(255,180,84,0.35);
    transition: transform 0.15s, box-shadow 0.15s;
  }
  button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(255,180,84,0.5);
  }
  button:active { transform: translateY(0); }
  #result {
    margin-top: 22px;
    padding: 18px;
    border-radius: 12px;
    background: rgba(255,180,84,0.08);
    border: 1px solid rgba(255,180,84,0.25);
    text-align: center;
    display: none;
  }
  #result.show { display: block; animation: fadeUp 0.4s ease; }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .result-label {
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 4px;
  }
  .result-price {
    font-family: 'Fraunces', serif;
    font-size: 34px;
    font-weight: 700;
    color: var(--amber-soft);
  }
</style>
</head>
<body>
  <div class="glow"></div>
  <div class="card">
    <div class="eyebrow">Instant Estimate</div>
    <h1>What's your house worth?</h1>
    <div class="sub">Fill in the details below and get an AI-powered price prediction.</div>

    <form id="predictForm">
      <div class="grid">
        <div class="field">
          <label>Area (sq ft)</label>
          <input type="number" id="Area" value="2000" required>
        </div>
        <div class="field">
          <label>Year Built</label>
          <input type="number" id="YearBuilt" value="2005" required>
        </div>
        <div class="field">
          <label>Bedrooms</label>
          <input type="number" id="Bedrooms" value="3" required>
        </div>
        <div class="field">
          <label>Bathrooms</label>
          <input type="number" id="Bathrooms" value="2" required>
        </div>
        <div class="field">
          <label>Floors</label>
          <input type="number" id="Floors" value="1" required>
        </div>
        <div class="field">
          <label>Garage</label>
          <select id="Garage">
            <option>Yes</option>
            <option>No</option>
          </select>
        </div>
        <div class="field full">
          <label>Location</label>
          <select id="Location">
            <option>Downtown</option>
            <option>Suburban</option>
            <option>Rural</option>
          </select>
        </div>
        <div class="field full">
          <label>Condition</label>
          <select id="Condition">
            <option>Excellent</option>
            <option>Good</option>
            <option>Fair</option>
          </select>
        </div>
      </div>

      <button type="submit">Predict Price →</button>
    </form>

    <div id="result">
      <div class="result-label">Estimated Price</div>
      <div class="result-price" id="priceValue">$0</div>
    </div>
  </div>

<script>
document.getElementById("predictForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    const btn = e.target.querySelector("button");
    const originalText = btn.innerText;
    btn.innerText = "Calculating...";
    btn.disabled = true;

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

    document.getElementById("priceValue").innerText =
        "$" + data.predicted_price.toLocaleString();
    document.getElementById("result").classList.add("show");

    btn.innerText = originalText;
    btn.disabled = false;
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