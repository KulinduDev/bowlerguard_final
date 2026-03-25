import pandas as pd
import numpy as np


def to_float_or_nan(value):
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def build_row(payload: dict, feature_cols) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    row = {}
    for col in feature_cols:
        row[col] = to_float_or_nan(payload.get(col, np.nan))

    x = pd.DataFrame([row], columns=feature_cols)

    if x.isna().all(axis=1).iloc[0]:
        raise ValueError("All features missing/invalid — enter at least some values.")

    return x