# src/features.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    coerce_binary_target,
    find_column,
    normalize_columns,
    safe_to_datetime,
    safe_divide,
)


def build_demographics_features(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Engineer customer-level demographic features.
    """
    df = normalize_columns(df)

    customer_col = find_column(df, ["customerid", "customer_id", "cust_id"])
    birthdate_col = find_column(df, ["birthdate", "date_of_birth", "dob"], required=False)

    out = df.copy()

    if birthdate_col is not None:
        out[birthdate_col] = safe_to_datetime(out[birthdate_col])
        ref_date = reference_date if reference_date is not None else pd.Timestamp.today().normalize()
        out["age_years"] = (ref_date - out[birthdate_col]).dt.days / 365.25
        out["birth_year"] = out[birthdate_col].dt.year
        out["birth_month"] = out[birthdate_col].dt.month
        out["birth_weekday"] = out[birthdate_col].dt.weekday
        out = out.drop(columns=[birthdate_col])

    # Keep other demographic columns as-is (categorical + numeric)
    # but avoid duplicate customer rows.
    out = out.drop_duplicates(subset=[customer_col]).reset_index(drop=True)
    return out


def build_perf_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Engineer loan-performance features and locate target column.
    """
    df = normalize_columns(df)

    customer_col = find_column(df, ["customerid", "customer_id", "cust_id"])
    target_col = find_column(df, ["good_bad_flag", "target", "label"], required=False)

    out = df.copy()

    approved_col = find_column(out, ["approveddate", "approved_date", "approvaldate"], required=False)
    creation_col = find_column(out, ["creationdate", "createdate", "created_at"], required=False)
    due_col = find_column(out, ["duedate", "due_date", "firstduedate"], required=False)
    close_col = find_column(out, ["closeddate", "close_date", "date_closed"], required=False)

    for c in [approved_col, creation_col, due_col, close_col]:
        if c is not None:
            out[c] = safe_to_datetime(out[c])

    if approved_col is not None:
        out["approved_year"] = out[approved_col].dt.year
        out["approved_month"] = out[approved_col].dt.month
        out["approved_weekday"] = out[approved_col].dt.weekday

    if creation_col is not None:
        out["creation_year"] = out[creation_col].dt.year
        out["creation_month"] = out[creation_col].dt.month
        out["creation_weekday"] = out[creation_col].dt.weekday

    if approved_col is not None and creation_col is not None:
        out["approval_delay_days"] = (out[approved_col] - out[creation_col]).dt.days

    # Keep raw date columns out of the model features
    for c in [approved_col, creation_col, due_col, close_col]:
        if c is not None and c in out.columns:
            out = out.drop(columns=[c])

    return out, target_col


def build_prevloan_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress trainprevloans.csv from many rows per customer into customer-level features.
    """
    df = normalize_columns(df)

    customer_col = find_column(df, ["customerid", "customer_id", "cust_id"])

    loannumber_col = find_column(df, ["loannumber", "loan_number", "loan_no"], required=False)
    loanamount_col = find_column(df, ["loanamount", "loan_amount", "amount"], required=False)
    due_col = find_column(df, ["duedate", "due_date", "firstduedate"], required=False)
    close_col = find_column(df, ["closeddate", "close_date", "date_closed"], required=False)
    approved_col = find_column(df, ["approveddate", "approved_date", "approvaldate"], required=False)
    term_col = find_column(df, ["termdays", "term_days", "loanterm", "termid"], required=False)

    out = df.copy()

    for c in [due_col, close_col, approved_col]:
        if c is not None:
            out[c] = safe_to_datetime(out[c])

    if due_col is not None and close_col is not None:
        out["days_to_repay"] = (out[close_col] - out[due_col]).dt.days
        out["late_repayment_flag"] = (out[close_col] > out[due_col]).astype(int)
    else:
        out["days_to_repay"] = np.nan
        out["late_repayment_flag"] = 0

    agg_dict = {
        "prevloan_count": (customer_col, "size"),
        "late_repayment_count": ("late_repayment_flag", "sum"),
        "late_repayment_rate": ("late_repayment_flag", "mean"),
        "days_to_repay_mean": ("days_to_repay", "mean"),
        "days_to_repay_median": ("days_to_repay", "median"),
        "days_to_repay_std": ("days_to_repay", "std"),
    }

    if loannumber_col is not None:
        agg_dict["max_loannumber"] = (loannumber_col, "max")
        agg_dict["unique_loannumber"] = (loannumber_col, "nunique")

    if loanamount_col is not None:
        agg_dict["prev_loanamount_mean"] = (loanamount_col, "mean")
        agg_dict["prev_loanamount_max"] = (loanamount_col, "max")
        agg_dict["prev_loanamount_sum"] = (loanamount_col, "sum")

    if term_col is not None:
        agg_dict["prev_term_mean"] = (term_col, "mean")
        agg_dict["prev_term_max"] = (term_col, "max")

    grouped = (
        out.groupby(customer_col)
        .agg(**agg_dict)
        .reset_index()
    )

    # Clean infinities if any
    grouped = grouped.replace([np.inf, -np.inf], np.nan)

    return grouped


def build_model_frame(
    demographics: pd.DataFrame,
    perf: pd.DataFrame,
    prevloans: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Merge all three files into one modeling table.
    """
    demographics = normalize_columns(demographics)
    perf = normalize_columns(perf)
    prevloans = normalize_columns(prevloans)

    perf_feat, target_col = build_perf_features(perf)
    reference_date = None

    # If there is a date in perf, use it as a reference for age calculation.
    for col in perf.columns:
        if "date" in col:
            reference_date = pd.to_datetime(perf[col], errors="coerce").max()
            if pd.notna(reference_date):
                break

    demo_feat = build_demographics_features(demographics, reference_date=reference_date)
    prev_feat = build_prevloan_features(prevloans)

    customer_col_demo = find_column(demo_feat, ["customerid", "customer_id", "cust_id"])
    customer_col_perf = find_column(perf_feat, ["customerid", "customer_id", "cust_id"])
    customer_col_prev = find_column(prev_feat, ["customerid", "customer_id", "cust_id"])

    merged = perf_feat.merge(
        demo_feat,
        how="left",
        left_on=customer_col_perf,
        right_on=customer_col_demo,
        suffixes=("", "_demo"),
    )

    merged = merged.merge(
        prev_feat,
        how="left",
        left_on=customer_col_perf,
        right_on=customer_col_prev,
        suffixes=("", "_prev"),
    )

    # Target
    if target_col is None:
        target_col = find_column(merged, ["good_bad_flag", "target", "label"], required=True)

    y = coerce_binary_target(merged[target_col])

    # Drop identifiers and target from features
    drop_cols = {
        target_col,
        customer_col_perf,
        customer_col_demo,
        customer_col_prev,
    }

    # Remove any duplicate key columns introduced by merge
    drop_cols.update([c for c in merged.columns if c.endswith("_demo") or c.endswith("_prev")])

    X = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    # Keep only reasonable feature values
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y
