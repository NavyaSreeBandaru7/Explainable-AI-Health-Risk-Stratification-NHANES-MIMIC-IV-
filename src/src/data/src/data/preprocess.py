import numpy as np
import pandas as pd

def build_label_diabetes(df: pd.DataFrame) -> pd.Series:
    """
    Diabetes label:
    - HbA1c (LBXGH) >= 6.5 OR
    - Self-reported diabetes diagnosis (DIQ010 == 1)
    """
    a1c = df.get("LBXGH")
    self_report = df.get("DIQ010")
    y = ((a1c >= 6.5) | (self_report == 1)).astype("int")
    return y

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    # Simple + defensible features for baseline healthcare risk model
    cols = [
        "RIDAGEYR",  # age
        "RIAGENDR",  # sex
        "RIDRETH3",  # race/ethnicity
        "DMDEDUC2",  # education (adults)
        "INDFMPIR",  # poverty income ratio
    ]
    return df[cols].copy()

def clean_inputs(X: pd.DataFrame) -> pd.DataFrame:
    # Replace common NHANES codes for "Refused/Don't know" with NaN
    X = X.replace({7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan})
    return X
