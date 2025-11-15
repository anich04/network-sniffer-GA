#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
import json

MODEL="src/ga_engine/ga_model.pkl"
DATA="src/data/processed/features.csv"
OUT="src/ga_engine/ga_rules.json"

def main():
    model = joblib.load(MODEL)
    df = pd.read_csv(DATA)

    # simple heuristic threshold builder
    rules = {"thresholds": {}}
    if "size_norm" in df.columns:
        rules["thresholds"]["size_norm"] = float(df["size_norm"].quantile(0.98))

    with open(OUT, "w") as f:
        json.dump(rules, f, indent=2)

    print("Wrote GA rules to", OUT)

if __name__ == "__main__":
    main()
