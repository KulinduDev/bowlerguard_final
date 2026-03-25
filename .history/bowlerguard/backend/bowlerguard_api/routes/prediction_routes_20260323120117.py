from flask import Blueprint, jsonify, request
from ..utils.auth import role_required
from ..utils.data_utils import build_row
from ..services.model_service import model_service

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.post("/predict")
@role_required(["admin", "coach", "physio", "analyst", "player"])
def predict():
    try:
        payload = request.get_json(force=True)
        x = build_row(payload, model_service.feature_cols)
        result = model_service.predict(x)

        return jsonify({
            "predicted_label": result["predicted_label"],
            "probabilities": result["probabilities"],
            "fatigue_score": result["fatigue_score"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@prediction_bp.post("/explain")
@role_required(["admin", "coach", "analyst"])
def explain():
    try:
        payload = request.get_json(force=True)
        x = build_row(payload, model_service.feature_cols)

        result = model_service.explain(x)

        return jsonify({
            "predicted_label": result["predicted_label"],
            "predicted_probabilities": result["predicted_probabilities"],
            "top_contributors": result["top_contributors"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400