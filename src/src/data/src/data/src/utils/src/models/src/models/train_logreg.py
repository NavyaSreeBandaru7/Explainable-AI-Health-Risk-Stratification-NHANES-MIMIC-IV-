import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR
from src.data.load_nhanes import load_and_merge
from src.data.preprocess import build_label_diabetes, select_features, clean_inputs
from src.utils.metrics import compute_metrics

def build_pipeline():
    num_cols = ["RIDAGEYR", "INDFMPIR"]
    cat_cols = ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=2000)
    return Pipeline([("pre", pre), ("clf", clf)])

def main():
    df = load_and_merge()
    y = build_label_diabetes(df)
    X = clean_inputs(select_features(df))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)
    print("LogReg metrics:", metrics)

    out = MODELS_DIR / "logreg_pipeline.joblib"
    joblib.dump({"model": model, "X_test": X_test, "y_test": y_test}, out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
