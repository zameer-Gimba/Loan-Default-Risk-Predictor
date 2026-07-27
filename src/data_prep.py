# src/data_prep.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.features import build_model_frame


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_raw_datasets(
    demographics_path: str | Path,
    perf_path: str | Path,
    prevloans_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demographics = load_csv(demographics_path)
    perf = load_csv(perf_path)
    prevloans = load_csv(prevloans_path)
    return demographics, perf, prevloans


def prepare_training_data(
    demographics_path: str | Path,
    perf_path: str | Path,
    prevloans_path: str | Path,
):
    demographics, perf, prevloans = load_raw_datasets(
        demographics_path, perf_path, prevloans_path
    )
    X, y = build_model_frame(demographics, perf, prevloans)
    return X, y


def prepare_inference_data(
    demographics_path: str | Path,
    perf_path: str | Path,
    prevloans_path: str | Path,
):
    """
    Same as training prep, but kept separate for clarity in the Streamlit app.
    """
    return prepare_training_data(demographics_path, perf_path, prevloans_path)
