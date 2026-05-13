# BowlerGuard

> Predicting immediate injury risk for cricket fast bowlers from accumulated fatigue — no sensors required.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

---

## What it does

BowlerGuard is a Sports Tech SaaS platform that uses machine learning
to flag when a fast bowler is at risk of injury — based purely on
workload, recovery, environment, and match context data available
from public cricket records. No wearables, no sensors.

The goal: give coaches and analysts an auditable, explainable tool
to make data-driven decisions about bowling loads before injuries happen.

---

## How it works

1. **ETL pipeline** — processes several hundred Cricsheet JSON files
   into ~5,000 player-match instances
2. **Feature engineering** — 12 features across workload, recovery,
   environment, and match context
3. **ML model** — multi-output XGBoost pipeline:
   - 3-class injury risk classifier
   - ROC-AUC: **0.85** | Accuracy: **68.77%**
   - Baseline (rule-only): 46.75% — a **22-point improvement**
4. **Explainability** — class-specific SHAP values make every
   prediction auditable; validated with active coaches and analysts
5. **API** — Flask MVP with a physiological simulation engine,
   architected for Docker containerisation

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.85 |
| Accuracy | 68.77% |
| Rule-only baseline | 46.75% |
| Improvement | +22 points |

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | Python, Pandas, Cricsheet JSON |
| ML | XGBoost, Scikit-learn, SHAP |
| API | Flask |
| Infra | Docker |

---



---

## Author

**Kulindu Ransika Hewamaddumage**
[LinkedIn](https://linkedin.com/in/https://https://www.linkedin.com/in/kulindu/) · [GitHub](https://github.com/KulinduDev)
