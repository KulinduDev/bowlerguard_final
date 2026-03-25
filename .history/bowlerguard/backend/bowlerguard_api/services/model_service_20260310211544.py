from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap


class ModelService:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[3]
        model_dir = project_root / "model_store"
        data_dir = project_root / "data" / "processed"

        # Load saved artefacts
        self.risk_model = joblib.load(model_dir / "risk_model.pkl")
        self.fatigue_model = joblib.load(model_dir / "fatigue_model.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        self.label_classes = joblib.load(model_dir / "risk_label_classes.pkl")

        # Split pipeline parts for easier use
        self.imputer = self.risk_model.named_steps["imputer"]
        self.xgb_model = self.risk_model.named_steps["model"]

        self.explainer = None
        self.explainer_error = None

        try:
            # Load a small background sample from training data
            X_train = pd.read_parquet(data_dir / "X_train.parquet")

            # Keep only deployed feature columns
            X_train = X_train[self.feature_cols].copy()

            # Small background sample for SHAP
            background = X_train.sample(
                n=min(100, len(X_train)),
                random_state=42
            )

            # Impute background because the underlying XGBoost model expects numeric values
            background_imp = self.imputer.transform(background)

            # Generic SHAP explainer using a callable instead of TreeExplainer
            self.explainer = shap.Explainer(
                self.xgb_model.predict_proba,
                background_imp
            )

        except Exception as e:
            self.explainer = None
            self.explainer_error = str(e)

    def predict(self, x: pd.DataFrame):
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

    def explain(self, x: pd.DataFrame):
        if self.explainer is None:
            raise ValueError(
                "SHAP explainer could not be initialized. "
                f"Details: {self.explainer_error}"
            )

        # Prediction first
        proba = self.risk_model.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self.label_classes[pred_idx]

        # Impute input before passing to the underlying XGBoost model
        x_imp = self.imputer.transform(x)

        # SHAP explanation on imputed array
        explanation = self.explainer(x_imp)

        # For multiclass: shape is usually (samples, features, classes)
        values = explanation.values

        if values.ndim == 3:
            contrib = values[0, :, pred_idx]
        elif values.ndim == 2:
            # Fallback in case SHAP returns single-output style
            contrib = values[0, :]
        else:
            raise ValueError("Unexpected SHAP output shape.")

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