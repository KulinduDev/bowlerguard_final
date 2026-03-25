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

        # load saved artefacts
        self.risk_model = joblib.load(model_dir / "risk_model.pkl")
        self.fatigue_model = joblib.load(model_dir / "fatigue_model.pkl")
        self.feature_cols = joblib.load(model_dir / "feature_cols.pkl")
        self.label_classes = joblib.load(model_dir / "risk_label_classes.pkl")

        # split pipeline parts for easier use
        self.imputer = self.risk_model.named_steps["imputer"]
        self.xgb_model = self.risk_model.named_steps["model"]

        self.explainer = None
        self.explainer_error = None

        try:
            # load a small background sample from training data
            X_train = pd.read_parquet(data_dir / "X_train.parquet")

            # keep only deployed feature columns
            X_train = X_train[self.feature_cols].copy()

            # small background sample for SHAP
            background = X_train.sample(
                n=min(100, len(X_train)),
                random_state=42
            )

            # impute background because the underlying model expects numeric values
            background_imp = self.imputer.transform(background)

            # generic SHAP explainer using a callable
            self.explainer = shap.Explainer(
                self.xgb_model.predict_proba,
                background_imp
            )

        except Exception as e:
            self.explainer = None
            self.explainer_error = str(e)

    def calibrated_label(self, proba: np.ndarray) -> str:
        """
        Deployment-time calibration layer.
        Uses probability thresholds instead of pure argmax so the UI
        shows more meaningful class separation.
        """
        low_p, med_p, high_p = [float(p) for p in proba]

        # high-risk override
        if high_p >= 0.20:
            return "High"

        # low-risk override
        if low_p >= 0.23:
            return "Low"

        # otherwise keep as Medium
        return "Medium"

    def predict(self, x: pd.DataFrame):
        proba = self.risk_model.predict_proba(x)[0]
        fatigue = float(self.fatigue_model.predict(x)[0])

        # use calibrated decision logic instead of raw argmax
        pred_label = self.calibrated_label(proba)

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

        # prediction first
        proba = self.risk_model.predict_proba(x)[0]
        pred_label = self.calibrated_label(proba)

        # for SHAP, keep argmax index to extract dominant probability class
        pred_idx = int(np.argmax(proba))

        # impute input before passing to the underlying model
        x_imp = self.imputer.transform(x)

        # sHAP explanation on imputed array
        explanation = self.explainer(x_imp)

        # for multiclass: shape is usually (samples, features, classes)
        values = explanation.values

        if values.ndim == 3:
            contrib = values[0, :, pred_idx]
        elif values.ndim == 2:
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