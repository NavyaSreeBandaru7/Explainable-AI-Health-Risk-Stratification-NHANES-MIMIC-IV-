# Model Card — NHANES Diabetes Risk (XAI + Fairness)

## Model
- Type: LightGBM classifier (tabular)
- Task: Predict diabetes risk
- Dataset: NHANES 2017–2018 (public CDC data)
- Label: HbA1c >= 6.5 OR self-reported diabetes

## Intended Use
- Educational/research demonstration of explainable and fair ML for healthcare risk stratification.
- Not for clinical decision-making.

## Metrics (fill after running)
- AUROC: [ ]
- AUPRC: [ ]
- Brier score: [ ]

## Explainability
- SHAP global feature importance
- SHAP local (single-patient) explanation waterfall plot

## Fairness Checks
- Group metrics by sex and race/ethnicity
- FPR/FNR differences reviewed

## Limitations
- NHANES is cross-sectional (not longitudinal).
- Label definition may miss undiagnosed cases.
- No clinical deployment validation.

## Ethics
- Public dataset, no PHI
- Bias audit included
