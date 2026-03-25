from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import shap

app = Flask(__name__)
CORS(app)


risk_model = joblib.load("../model_store/risk_model.pkl")
fatigue_model = joblib.load("../model_store/fatigue_model.pkl")
feature_cols = joblib.load("../model_store/feature_cols.pkl")
label_classes = joblib.load("../model_store/risk_label_classes.pkl")


if isinstance(label_classes, (pd.Series, pd.Index, np.ndarray)):
    label_classes = list(label_classes)
else:
    label_classes = list(label_classes)

label_classes = [str(x) for x in label_classes]

explainer = shap.TreeExplainer(risk_model)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def validate_and_build_row(payload: dict) -> pd.DataFrame:
    """
    Validates incoming JSON payload and returns a single-row DataFrame
    with columns in exactly the feature_cols order.
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object (dictionary).")

    missing = [c for c in feature_cols if c not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    
    row = {c: payload[c] for c in feature_cols}

    
    x = pd.DataFrame([row], columns=feature_cols)

   
    for c in feature_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    
    bad = [c for c in feature_cols if pd.isna(x.loc[0, c])]
    if bad:
        raise ValueError(f"Non-numeric or invalid values for: {bad}")

    return x


@app.post("/predict")
def predict():
    """
    Returns:
      - predicted_label (Low/Medium/High)
      - probabilities per class
      - fatigue_score (continuous)
    """
    try:
        payload = request.get_json(force=True)
        x = validate_and_build_row(payload)

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
def explain():
    """
    Minimal SHAP explanation for ONE sample:
    Returns top feature contributions for the predicted class.
    """
    try:
        payload = request.get_json(force=True)
        x = validate_and_build_row(payload)

        proba = risk_model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = label_classes[pred_idx]

        shap_values = explainer.shap_values(x)

        
        
        if isinstance(shap_values, list):
            contrib = np.array(shap_values[pred_idx])[0]  # (n_features,)

        else:
            sv = np.array(shap_values)

            
            
            if sv.ndim == 3 and sv.shape[0] == 1 and sv.shape[1] == len(feature_cols):
                
                contrib = sv[0, :, pred_idx]

            elif sv.ndim == 3 and sv.shape[1] == len(feature_cols) and sv.shape[2] == len(label_classes):
                
                contrib = sv[0, :, pred_idx]

            elif sv.ndim == 3 and sv.shape[0] == len(label_classes) and sv.shape[2] == len(feature_cols):
                
                contrib = sv[pred_idx, 0, :]

            elif sv.ndim == 2 and sv.shape[1] == len(feature_cols):
                
                contrib = sv[0, :]

            else:
                raise ValueError(f"Unexpected SHAP values shape: {sv.shape}")

        contrib = np.array(contrib).reshape(-1)

        pairs = list(zip(feature_cols, contrib))
        pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)

        top_k = 10
        top = [{"feature": f, "shap_value": float(v)} for f, v in pairs_sorted[:top_k]]

        return jsonify({
            "predicted_label": pred_label,
            "predicted_probabilities": {label_classes[i]: float(proba[i]) for i in range(len(label_classes))},
            "top_contributors": top
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000, debug=True)
