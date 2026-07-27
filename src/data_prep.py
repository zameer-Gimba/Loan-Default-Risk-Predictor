# src/data_prep.py

from __future__ import annotations

from typing import Tuple, Any
import pandas as pd

from src.features import build_model_frame


def load_csv(path_or_buffer: Any) -> pd.DataFrame:
    """
    Loads a CSV file. Accepts local string/Path file paths or 
    Streamlit uploaded file buffer streams.
    """
    return pd.read_csv(path_or_buffer)


def load_raw_datasets(
    demographics_path: Any,
    perf_path: Any,
    prevloans_path: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all three required datasets from paths or file buffers."""
    demographics = load_csv(demographics_path)
    perf = load_csv(perf_path)
    prevloans = load_csv(prevloans_path)
    return demographics, perf, prevloans


def prepare_training_data(
    demographics_path: Any,
    perf_path: Any,
    prevloans_path: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads raw datasets and transforms them into feature and target matrices."""
    demographics, perf, prevloans = load_raw_datasets(
        demographics_path, perf_path, prevloans_path
    )
    X, y = build_model_frame(demographics, perf, prevloans)
    return X, y


def prepare_inference_data(
    demographics_path: Any,
    perf_path: Any,
    prevloans_path: Any,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Same as training prep, but kept separate for clarity in the Streamlit app.
    Accepts file buffers directly from the user interface.
    """
    return prepare_training_data(demographics_path, perf_path, prevloans_path)
