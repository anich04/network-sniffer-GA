#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib

def apply_ga(df, model):
    thr = model.get("thresholds", {})
    size_t = thr.get("size_norm", 3.0)
    df["suspicious"] = df.get("size_norm", 0) > size_t
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="src/data/processed/features.csv")
    parser.add_argument("--model", default="src/ga_engine/ga_model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    model = joblib.load(args.model)
    df = apply_ga(df, model)
    df.to_csv("src/data/processed/results.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
