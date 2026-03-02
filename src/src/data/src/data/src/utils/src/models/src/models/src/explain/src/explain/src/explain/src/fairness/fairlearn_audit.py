import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate
from sklearn.metrics import roc_auc_score

from src.config import MODELS_DIR

def main():
    bundle = joblib.load(MODELS_DIR / "lgbm_pipeline.joblib")
    model = bundle["model"]
    X_test = bundle["X_test"].copy()
    y_test = bundle["y_test"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Sensitive features (in raw space)
    # RIAGENDR: 1=Male, 2=Female
    # RIDRETH3: race/ethnicity categories
    sensitive = pd.DataFrame({
        "sex": X_test["RIAGENDR"].astype("Int64"),
        "race_eth": X_test["RIDRETH3"].astype("Int64"),
    })

    metrics = {
        "selection_rate": selection_rate,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "auroc": lambda yt, yp: roc_auc_score(yt, yp),
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    print("\n=== Overall ===")
    print(mf.overall)

    print("\n=== By Sex ===")
    print(mf.by_group["selection_rate"].groupby(level=0).mean())
    print(mf.by_group["fpr"].groupby(level=0).mean())
    print(mf.by_group["fnr"].groupby(level=0).mean())

    # AUROC requires probability; compute separately by group
    for group_name in ["sex", "race_eth"]:
        print(f"\n=== AUROC by {group_name} ===")
        for g in sorted(sensitive[group_name].dropna().unique()):
            mask = (sensitive[group_name] == g).to_numpy()
            if mask.sum() < 50:
                continue
            print(g, roc_auc_score(np.array(y_test)[mask], y_prob[mask]))

if __name__ == "__main__":
    main()
