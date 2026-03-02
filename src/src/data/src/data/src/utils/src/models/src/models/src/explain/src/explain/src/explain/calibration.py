import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.config import MODELS_DIR, REPORTS_DIR
from src.utils.metrics import compute_metrics

def main():
    bundle = joblib.load(MODELS_DIR / "lgbm_pipeline.joblib")
    model = bundle["model"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    print("Calibration metrics (includes Brier):", metrics)

    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")

    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    out = REPORTS_DIR / "calibration_curve.png"
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
