from flask import Flask, request, jsonify, session
from flask_cors import CORS
from functools import wraps
import joblib
import pandas as pd
import numpy as np
import shap

app = Flask(__name__)
app.secret_key = "bowlerguard_secret"
CORS(app, supports_credentials=True)

USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "coach": {"password": "coach123", "role": "coach"},
    "physio": {"password": "physio123", "role": "physio"},
    "analyst": {"password": "analyst123", "role": "analyst"},
    "player": {"password": "player123", "role": "player"}
}

risk_model = joblib.load("../model_store/risk_model.pkl")
fatigue_model = joblib.load("../model_store/fatigue_model.pkl")
feature_cols = joblib.load("../model_store/feature_cols.pkl")
label_classes = joblib.load("../model_store/risk_label_classes.pkl")

xgb_model = risk_model.named_steps["model"]
explainer = shap.TreeExplainer(xgb_model)


def role_required(allowed_roles):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "role" not in session:
                return jsonify({"error": "Unauthorized"}), 401
            if session["role"] not in allowed_roles:
                return jsonify({"error": "Access denied"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = USERS.get(username)
    if user and user["password"] == password:
        session["user"] = username
        session["role"] = user["role"]
        return jsonify({"message": "Login successful", "role": user["role"]})
    return jsonify({"error": "Invalid credentials"}), 401


def _to_float_or_nan(v):
    if v is None:
        return np.nan
    if isinstance(v, str) and v.strip() == "":
        return np.nan
    try:
        return float(v)
    except:
        return np.nan


def build_row(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    row = {}
    for c in feature_cols:
        row[c] = _to_float_or_nan(payload.get(c, np.nan))

    x = pd.DataFrame([row], columns=feature_cols)
    if x.isna().all(axis=1).iloc[0]:
        raise ValueError("All features missing/invalid — enter at least some values.")
    return x


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)
        x = build_row(payload)

        proba = risk_model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = label_classes[pred_idx]

        fatigue = float(fatigue_model.predict(x)[0])

        return jsonify({
            "predicted_label": pred_label,
            "probabilities": {label_classes[i]: float(proba[i]) for i in range(len(label_classes))},
            "fatigue_score": fatigue
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/explain")
@role_required(["admin", "coach", "analyst"])
def explain():
    try:
        payload = request.get_json(force=True)
        x = build_row(payload)

        proba = risk_model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = label_classes[pred_idx]

        x_imp = risk_model.named_steps["imputer"].transform(x)
        shap_vals = explainer.shap_values(x_imp)

        contrib = shap_vals[0, :, pred_idx]
        pairs = list(zip(feature_cols, contrib))
        pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)

        top = [{"feature": f, "shap_value": float(v)} for f, v in pairs_sorted[:10]]

        return jsonify({
            "predicted_label": pred_label,
            "predicted_probabilities": {label_classes[i]: float(proba[i]) for i in range(len(label_classes))},
            "top_contributors": top
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)