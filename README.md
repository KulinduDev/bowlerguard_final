# BowlerGuard

BowlerGuard is an AI/ML-based decision-support prototype for predicting fatigue-driven injury risk in cricket fast bowlers using workload, recovery, and environmental context. The system was developed as a final year research project and focuses on Sri Lankan Test-match fast bowlers under local playing conditions.

The project combines:
- publicly available workload-related match data
- literature-based proxy fatigue and injury-risk generation
- machine learning for multiclass injury-risk prediction
- explainable AI for identifying the main drivers behind each prediction
- a web-based prototype with authentication and role-based access

---

## Project Aim

The aim of BowlerGuard is to provide a practical and explainable framework for short-term injury-risk assessment when direct physiological and medical injury datasets are not available.

Instead of relying on expensive real-time wearable data, the system uses:
- recent bowling workload
- recovery period
- match context
- weather-related stress indicators

to estimate:
- a fatigue score
- an injury-risk class: **Low**, **Medium**, or **High**
- the top contributing factors behind the prediction

---

## Research Motivation

Fast bowlers experience repeated high-intensity loading, especially in multi-day formats such as Test cricket. In Sri Lankan playing conditions, heat and humidity can further amplify accumulated fatigue. However, real injury datasets and continuous physiological monitoring data are rarely available in public cricket analytics settings.

BowlerGuard addresses this gap by:
1. generating proxy fatigue and proxy injury-risk labels using literature-based rules
2. training machine learning models on the resulting dataset
3. deploying the selected models in a working prototype for authenticated users such as coaches, physiotherapists, and analysts

---

## Key Features

- Login-based access to the system
- Role-aware interface behaviour
- Input form for workload, recovery, and environmental variables
- Injury-risk prediction with class probabilities
- Fatigue score prediction
- SHAP-based explanation for model predictions
- Offline notebook pipeline for:
  - proxy-label generation
  - data audit
  - feature engineering
  - model training
  - model evaluation
  - explainability analysis

---

## System Overview

The project has two main layers:

### 1. Offline ML Pipeline
Implemented in Jupyter notebooks for:
- dataset preparation
- proxy target generation
- train/test splitting
- feature engineering
- model comparison
- final model selection
- artifact saving

### 2. Online Prototype
Implemented using:
- **Flask** backend
- **HTML/CSS/JavaScript** frontend

The backend loads saved model artifacts and exposes routes for:
- session checking
- login/logout
- prediction
- explanation

---

## Project Structure

```text
bowlerguard/
├── backend/
│   ├── app.py
│   └── bowlerguard_api/
│       ├── __init__.py
│       ├── auth_store.py
│       ├── extensions.py
│       ├── routes/
│       │   ├── auth_routes.py
│       │   ├── core_routes.py
│       │   └── prediction_routes.py
│       ├── services/
│       │   └── model_service.py
│       └── utils/
│           ├── auth.py
│           └── data_utils.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
├── model_store/
│   ├── risk_model.pkl
│   ├── fatigue_model.pkl
│   ├── feature_cols.pkl
│   └── risk_label_classes.pkl
├── notebooks/
│   ├── 00_proxy_label_refresh.ipynb
│   ├── 01_data_audit.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_final_models_save.ipynb
│   └── 05_explainability.ipynb
└── src/
    └── bowlerguard/
        └── proxy_rules.py
