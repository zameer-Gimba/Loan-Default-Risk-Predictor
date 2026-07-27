# src/utils.py

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the dataframe with normalized column names.
    """
    out = df.copy()
    out.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        for c in out.columns
    ]
    return out


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    """
    Find the first matching column from a list of candidate names.
    Matching is done after normalization.
    """
    normalized_map = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized_map:
            return normalized_map[key]
    if required:
        raise KeyError(
            f"None of these columns were found: {list(candidates)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def coerce_binary_target(y: pd.Series) -> pd.Series:
    """
    Convert common binary target formats to 0/1.
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.fillna(0).astype(int)

    s = y.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "bad": 1,
        "good": 0,
        "default": 1,
        "non_default": 0,
        "non-default": 0,
        "delinquent": 1,
        "paid": 0,
    }
    return s.map(mapping).fillna(0).astype(int)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator not in (0, 0.0, None) else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the main binary classification metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "specificity": safe_divide(tn, tn + fp),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_negative_rate": safe_divide(fn, fn + tp),
    }
    return metrics


def best_threshold_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Find the probability threshold that maximizes F1-score.
    Returns (best_threshold, best_f1).
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)

    f1_scores = 2 * (precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def save_bundle(bundle: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> Dict[str, Any]:
    return joblib.load(path)


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
