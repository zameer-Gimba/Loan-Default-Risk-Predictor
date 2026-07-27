# src/train.py

from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_prep import prepare_training_data
from src.utils import (
    best_threshold_from_probs,
    compute_metrics,
    ensure_directory,
    save_bundle,
)

warnings.filterwarnings("ignore")


RANDOM_STATE = 42


def make_one_hot_encoder():
    """
    Compatibility helper for different scikit-learn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(X: pd.DataFrame, scale_pos_weight: float) -> Pipeline:
    preprocessor = build_preprocessor(X)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipe


def train_model(
    demographics_path: str | Path,
    perf_path: str | Path,
    prevloans_path: str | Path,
    model_dir: str | Path = "models",
):
    X, y = prepare_training_data(demographics_path, perf_path, prevloans_path)

    # Holdout split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    pipe = build_pipeline(X_train, scale_pos_weight=scale_pos_weight)

    # 5-fold stratified CV on the training split only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv_scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    # OOF probabilities for threshold tuning
    oof_probs = cross_val_predict(
        pipe,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    best_threshold, best_oof_f1 = best_threshold_from_probs(y_train.values, oof_probs)

    # Fit final model on all training data
    pipe.fit(X_train, y_train)

    test_probs = pipe.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    test_metrics = compute_metrics(y_test.values, test_preds, test_probs)

    cv_summary = {
        key.replace("test_", ""): {
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
        }
        for key, value in cv_scores.items()
        if key.startswith("test_")
    }

    bundle = {
        "pipeline": pipe,
        "threshold": float(best_threshold),
        "cv_summary": cv_summary,
        "oof_f1": float(best_oof_f1),
        "test_metrics": test_metrics,
        "feature_count": int(X.shape[1]),
        "feature_names": list(X.columns),
        "random_state": RANDOM_STATE,
    }

    model_dir = ensure_directory(model_dir)
    bundle_path = Path(model_dir) / "loan_default_risk_model.joblib"
    save_bundle(bundle, bundle_path)

    # Save a readable metrics summary too
    metrics_path = Path(model_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": bundle["threshold"],
                "oof_f1": bundle["oof_f1"],
                "cv_summary": bundle["cv_summary"],
                "test_metrics": bundle["test_metrics"],
            },
            f,
            indent=2,
        )

    print("\nTraining completed.")
    print(f"Model saved to: {bundle_path}")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"OOF F1: {best_oof_f1:.4f}")
    print("\nCross-validation summary:")
    for metric, stats in cv_summary.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nHoldout test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nClassification report (holdout):")
    print(classification_report(y_test, test_preds, digits=4))

    return bundle


if __name__ == "__main__":
    train_model(
        demographics_path="data/traindemographics.csv",
        perf_path="data/trainperf.csv",
        prevloans_path="data/trainprevloans.csv",
        model_dir="models",
    )
