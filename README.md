# Explainable-AI-Health-Risk-Stratification-NHANES-MIMIC-IV

This project builds an **Explainable AI (XAI)** pipeline for healthcare risk prediction using **NHANES (public CDC data)** now, and is designed to be extended to **MIMIC-IV (PhysioNet)** after access approval.

## What this repo does
- Predicts **diabetes risk** using NHANES 2017–2018
- Uses **LightGBM** + baseline Logistic Regression
- Adds **SHAP explainability** (global + patient-level)
- Audits **fairness** across demographics using **Fairlearn**
- Validates **calibration** (reliability curve + Brier score)
- Produces governance-ready documentation (model card + checklist)

## Data
- NHANES is public and downloaded automatically from CDC endpoints (no data is committed).
- MIMIC-IV is free but requires PhysioNet credentialing and approval.

See: `DATA_ACCESS.md`

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
