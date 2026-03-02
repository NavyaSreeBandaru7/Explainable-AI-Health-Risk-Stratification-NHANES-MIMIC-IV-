import joblib
import shap
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, REPORTS_DIR

def main():
    bundle = joblib.load(MODELS_DIR / "lgbm_pipeline.joblib")
    model = bundle["model"]
    X_test = bundle["X_test"]

    pre = model.named_steps["pre"]
    X_proc = pre.transform(X_test)

    oh = pre.named_transformers_["cat"].named_steps["oh"]
    cat_features = oh.get_feature_names_out(pre.transformers_[1][2])
    num_features = pre.transformers_[0][2]
    feature_names = list(num_features) + list(cat_features)

    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_proc)

    # Explain one sample (first row)
    i = 0
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(values=shap_values[i], base_values=explainer.expected_value,
                         data=X_proc[i].toarray()[0] if hasattr(X_proc[i], "toarray") else X_proc[i],
                         feature_names=feature_names),
        show=False
    )
    out = REPORTS_DIR / "shap_local_waterfall.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
