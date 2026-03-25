from pathlib import Path
import joblib
import numpy as np
import shap


class ModelService:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]
        model_dir = project_root / "model_store"

        self.risk_model = joblib.load(model_dir / "risk_model.pkl")
        self.fatigue_model = joblib.load(model_dir / "fatigue_model.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        self.label_classes = joblib.load(model_dir / "risk_label_classes.pkl")

        self.explainer = None
        self.explainer_error = None

        try:
            xgb_model = self.risk_model.named_steps["model"]
            self.explainer = shap.TreeExplainer(xgb_model)
        except Exception as e:
            self.explainer = None
            self.explainer_error = str(e)

    def predict(self, x):
        proba = self.risk_model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.label_classes[pred_idx]
        fatigue = float(self.fatigue_model.predict(x)[0])

        return {
            "predicted_label": pred_label,
            "probabilities": {
                self.label_classes[i]: float(proba[i])
                for i in range(len(self.label_classes))
            },
            "fatigue_score": fatigue
        }

    def explain(self, x):
        if self.explainer is None:
            raise ValueError(
                "SHAP explainer could not be initialized for this saved model. "
                f"Details: {self.explainer_error}"
            )

        proba = self.risk_model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.label_classes[pred_idx]

        x_imp = self.risk_model.named_steps["imputer"].transform(x)
        shap_vals = self.explainer.shap_values(x_imp)

        contrib = shap_vals[0, :, pred_idx]
        pairs = list(zip(self.feature_cols, contrib))
        pairs_sorted = sorted(pairs, key=lambda t: abs(t[1]), reverse=True)

        top = [
            {"feature": f, "shap_value": float(v)}
            for f, v in pairs_sorted[:10]
        ]

        return {
            "predicted_label": pred_label,
            "predicted_probabilities": {
                self.label_classes[i]: float(proba[i])
                for i in range(len(self.label_classes))
            },
            "top_contributors": top
        }


model_service = ModelService()