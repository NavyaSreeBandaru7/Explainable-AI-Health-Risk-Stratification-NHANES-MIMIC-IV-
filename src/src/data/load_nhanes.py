import os
import requests
import pandas as pd
import pyreadstat

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"  # 2017–2018 cycle
FILES = {
    "DEMO_J": f"{BASE}/DEMO_J.xpt",  # demographics
    "DIQ_J":  f"{BASE}/DIQ_J.xpt",   # diabetes questionnaire
    "GHB_J":  f"{BASE}/GHB_J.xpt",   # HbA1c
}

def download_xpt(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def read_xpt(path: str) -> pd.DataFrame:
    df, _meta = pyreadstat.read_xport(path)
    return df

def load_and_merge(data_dir="data/raw") -> pd.DataFrame:
    local = {}
    for k, url in FILES.items():
        out = os.path.join(data_dir, f"{k}.xpt")
        download_xpt(url, out)
        local[k] = read_xpt(out)

    # Merge on SEQN (participant ID)
    df = local["DEMO_J"].merge(local["DIQ_J"], on="SEQN", how="left")
    df = df.merge(local["GHB_J"], on="SEQN", how="left")
    return df
