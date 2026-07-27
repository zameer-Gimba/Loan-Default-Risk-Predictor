# src/predict.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_prep import prepare_inference_data
from src.utils import load_bundle


def predict_from_raw_files(
    demographics_path: str | Path,
    perf_path: str | Path,
    prevloans_path: str | Path,
    model_path: str | Path = "models/loan_default_risk_model.joblib",
) -> pd.DataFrame:
    bundle = load_bundle(model_path)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    X, y = prepare_inference_data(demographics_path, perf_path, prevloans_path)

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = pd.DataFrame(
        {
            "default_probability": probs,
            "predicted_default": preds,
            "risk_label": ["High Risk" if p == 1 else "Low Risk" for p in preds],
        }
    )

    if len(y) == len(result):
        result["actual_target"] = y.values

    return result


def predict_from_dataframe(
    df: pd.DataFrame,
    model_path: str | Path = "models/loan_default_risk_model.joblib",
) -> pd.DataFrame:
    """
    Use when you already have a prepared feature dataframe.
    """
    bundle = load_bundle(model_path)
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    return pd.DataFrame(
        {
            "default_probability": probs,
            "predicted_default": preds,
            "risk_label": ["High Risk" if p == 1 else "Low Risk" for p in preds],
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Loan Default Risk Predictor")
    parser.add_argument("--demographics", type=str, help="Path to traindemographics.csv")
    parser.add_argument("--perf", type=str, help="Path to trainperf.csv")
    parser.add_argument("--prevloans", type=str, help="Path to trainprevloans.csv")
    parser.add_argument("--model", type=str, default="models/loan_default_risk_model.joblib")
    parser.add_argument("--output", type=str, default="predictions.csv")

    args = parser.parse_args()

    if not (args.demographics and args.perf and args.prevloans):
        raise SystemExit("Please provide --demographics, --perf, and --prevloans paths.")

    result = predict_from_raw_files(
        demographics_path=args.demographics,
        perf_path=args.perf,
        prevloans_path=args.prevloans,
        model_path=args.model,
    )

    result.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
