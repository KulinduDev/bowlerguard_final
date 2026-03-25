from __future__ import annotations

import numpy as np
import pandas as pd


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_esi_norm(temp_c: float, humidity_pct: float) -> float:
    """
    Simple Environmental Stress Index proxy: normalized 0..1.
    """
    if np.isnan(temp_c) or np.isnan(humidity_pct):
        return np.nan

    t = (temp_c - 22.0) / (36.5 - 22.0)
    h = (humidity_pct - 45.0) / (95.0 - 45.0)
    return clamp(0.6 * t + 0.4 * h, 0.0, 1.0)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def generate_proxy_targets_v3(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    v3 Proxy target generator:
    - Adds latent risk + latent recovery + random noise
    - Uses workload + weather + recovery to generate:
        * fatigue_score_0_100_v3
        * injury_risk_prob_v3
        * injury_risk_label_v3
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    required = [
        "deliveries_match", "deliveries_7d", "deliveries_28d",
        "acwr_std", "days_since_prev", "match_day",
        "avg_temp_c", "avg_humidity_pct",
        "inferred_fielding_time_minutes"
    ]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Missing required column for proxy generation: {c}")

    if "esi_norm" not in out.columns:
        out["esi_norm"] = np.nan

    mask_esi = out["esi_norm"].isna()
    out.loc[mask_esi, "esi_norm"] = out.loc[mask_esi].apply(
        lambda r: compute_esi_norm(r["avg_temp_c"], r["avg_humidity_pct"]), axis=1
    )

    # Latent unobserved factors
    if "player_id" in out.columns:
        players = out["player_id"].astype(str).unique()
        latent_risk = {p: rng.normal(0.0, 0.6) for p in players}
        latent_recovery = {p: rng.normal(0.0, 0.6) for p in players}
        lr = out["player_id"].astype(str).map(latent_risk).astype(float)
        lrec = out["player_id"].astype(str).map(latent_recovery).astype(float)
    else:
        lr = rng.normal(0.0, 0.6, size=len(out))
        lrec = rng.normal(0.0, 0.6, size=len(out))

    # Noise
    eps_fatigue = rng.normal(0.0, 4.0, size=len(out))
    eps_risk = rng.normal(0.0, 0.25, size=len(out))

    # Core signals
    deliveries_match = out["deliveries_match"].astype(float)
    deliveries_7d = out["deliveries_7d"].astype(float)
    deliveries_28d = out["deliveries_28d"].astype(float)
    acwr = out["acwr_std"].astype(float)
    rest_days = out["days_since_prev"].astype(float).fillna(0.0).clip(lower=0.0)
    match_day = out["match_day"].astype(float)
    esi = out["esi_norm"].astype(float)
    fielding = out["inferred_fielding_time_minutes"].astype(float)

    # Fatigue score
    fatigue_raw = (
        0.12 * deliveries_match +
        0.035 * deliveries_7d +
        6.0 * np.maximum(acwr - 1.0, 0.0) +
        14.0 * esi +
        9.0 * (1.0 / (1.0 + rest_days)) +
        0.012 * fielding +
        1.0 * np.maximum(match_day - 2.0, 0.0) -
        4.0 * lrec +
        eps_fatigue
    )

    fatigue_score = np.clip(fatigue_raw, 0.0, 100.0)
    out["fatigue_score_0_100_v3"] = fatigue_score

    # Injury risk probability - calibrated softer
    interaction = esi * np.maximum(acwr, 0.0)

    z = (
        -4.2 +
        0.022 * fatigue_score +
        0.30 * np.maximum(acwr - 1.0, 0.0) +
        0.35 * interaction +
        0.10 * (1.0 / (1.0 + rest_days)) +
        0.05 * (deliveries_match / 100.0) +
        0.06 * (deliveries_7d / 300.0) +
        0.05 * np.maximum(match_day - 2.0, 0.0) +
        0.08 * lr +
        eps_risk
    )

    risk_prob = sigmoid(z)
    out["injury_risk_prob_v3"] = risk_prob

    def prob_to_label(p: float) -> str:
        if p < 0.35:
            return "Low"
        elif p < 0.65:
            return "Medium"
        else:
            return "High"

    out["injury_risk_label_v3"] = out["injury_risk_prob_v3"].apply(prob_to_label)

    return out