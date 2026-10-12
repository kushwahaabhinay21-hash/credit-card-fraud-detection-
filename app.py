"""
Credit Card Fraud Detection - demo interface (Flask)
Run train.py first (it creates models/best_model.joblib), then:
    python app.py
and open http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "best_model.joblib")
SAMPLES_PATH = os.path.join(BASE, "models", "sample_transactions.json")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("models/best_model.joblib not found. Run:  python train.py --data data/creditcard.csv")

bundle = joblib.load(MODEL_PATH)
PIPE, THRESHOLD, FEATURES = bundle["pipeline"], bundle["threshold"], bundle["features"]
SAMPLES = json.load(open(SAMPLES_PATH)) if os.path.exists(SAMPLES_PATH) else []

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit Card Fraud Detection</title>
<style>
  body{font-family:Arial,sans-serif;max-width:760px;margin:30px auto;padding:0 14px;color:#222}
  textarea{width:100%;height:130px;font-family:monospace;font-size:13px}
  button{padding:8px 14px;margin:4px 4px 4px 0;cursor:pointer}
  .box{padding:14px;border-radius:8px;margin-top:16px;font-size:18px}
  .fraud{background:#fde2e2;border:1px solid #d33}
  .legit{background:#e1f5e1;border:1px solid #3a3}
  .err{background:#fff3cd;border:1px solid #cc9a06}
  small{color:#666}
</style></head><body>
<h2>Credit Card Fraud Detection</h2>
<p><small>Model: {{ model_name }} ({{ scenario }}) &middot; decision threshold: {{ threshold }}<br>
Academic prototype - not a production banking system.</small></p>
<form method="post">
  <p>Enter {{ n }} comma-separated values in this order:<br><small>{{ features }}</small></p>
  <textarea name="values" id="values">{{ values }}</textarea><br>
  {% for s in samples %}
    <button type="button" onclick="fill({{ loop.index0 }})">Load sample {{ loop.index }} ({{ s.label }})</button>
  {% endfor %}
  <br><button type="submit"><b>Predict</b></button>
</form>
{% if error %}<div class="box err">{{ error }}</div>{% endif %}
{% if result %}
  <div class="box {{ 'fraud' if result.is_fraud else 'legit' }}">
    <b>{{ result.label }}</b><br>
    Fraud probability: {{ '%.2f'|format(result.probability * 100) }}%
  </div>
{% endif %}
<script>
  const samples = {{ samples|tojson }};
  function fill(i){ document.getElementById('values').value = samples[i].values.join(', '); }
</script>
</body></html>
"""


def predict(values):
    if len(values) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} values, got {len(values)}.")
    row = pd.DataFrame([values], columns=FEATURES)
    prob = float(PIPE.predict_proba(row)[0, 1])
    is_fraud = prob >= THRESHOLD
    return {"probability": prob, "is_fraud": bool(is_fraud),
            "label": "Potentially Fraudulent" if is_fraud else "Legitimate"}


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    raw = ""
    if request.method == "POST":
        raw = request.form.get("values", "")
        try:
            values = [float(v) for v in raw.replace("\n", ",").split(",") if v.strip()]
            result = predict(values)
        except ValueError as e:
            error = f"Invalid input: {e}"
    return render_template_string(
        PAGE, result=result, error=error, values=raw, samples=SAMPLES,
        n=len(FEATURES), features=", ".join(FEATURES), threshold=THRESHOLD,
        model_name=bundle["model_name"], scenario=bundle["scenario"])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: POST {"values": [..30 numbers..]}"""
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict([float(v) for v in data.get("values", [])]))
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False)
