import joblib
import shap
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, REPORTS_DIR

def main():
    bundle = joblib.load(MODELS_DIR / "lgbm_pipeline.joblib")
    model = bundle["model"]
    X_test = bundle["X_test"]

    # Transform data to model input space
    X_proc = model.named_steps["pre"].transform(X_test)

    # Get feature names after one-hot encoding
    pre = model.named_steps["pre"]
    oh = pre.named_transformers_["cat"].named_steps["oh"]
    cat_features = oh.get_feature_names_out(pre.transformers_[1][2])
    num_features = pre.transformers_[0][2]
    feature_names = list(num_features) + list(cat_features)

    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_proc)

    plt.figure()
    shap.summary_plot(shap_values, X_proc, feature_names=feature_names, show=False)
    out = REPORTS_DIR / "shap_summary.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
