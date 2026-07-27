# app/streamlit_app.py
from __future__ import annotations
import os
import sys
from pathlib import Path

# This forces Python to look at the root repository folder
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data_prep import prepare_inference_data
from src.utils import load_bundle

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💳",
    layout="wide",
)

MODEL_PATH = Path("models/loan_default_risk_model.joblib")


@st.cache_resource
def load_model_bundle():
    return load_bundle(MODEL_PATH)


def main():
    st.title("Loan Default Risk Predictor")
    st.caption("3MTT Final Project | DSN Loan Default Prediction dataset | LightGBM")

    if not MODEL_PATH.exists():
        st.warning("Model file not found. Train the model first and save it in `models/loan_default_risk_model.joblib`.")
        st.stop()

    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.subheader("Data Source Status")
        
        # Configure file paths to point directly to your GitHub repository storage
        demo_path = Path("data/traindemographics.csv")
        perf_path = Path("data/trainperf.csv")
        prev_path = Path("data/trainprevloans.csv")

        # Verify that all three CSV files are saved in your data/ folder
        if demo_path.exists() and perf_path.exists() and prev_path.exists():
            st.success("🔄 Repository datasets detected in 'data/' folder successfully!")
            run_btn = st.button("Generate Predictions from Repository Data", type="primary")
        else:
            st.error("Missing raw CSV files in your repository 'data/' folder. Please ensure they are uploaded to GitHub.")
            st.info("Expected files: data/traindemographics.csv, data/trainperf.csv, data/trainprevloans.csv")
            run_btn = False

    with col2:
        st.subheader("Model Summary")
        st.write(f"**Threshold:** {threshold:.4f}")
        st.write(f"**Feature count:** {bundle.get('feature_count', 'N/A')}")
        st.write("**Target:** good_bad_flag")
        if "test_metrics" in bundle:
            st.json(bundle["test_metrics"])

    if run_btn:
        with st.spinner("Processing repository data and generating predictions..."):
            X, y = prepare_inference_data(
                demographics_path=demo_path,
                perf_path=perf_path,
                prevloans_path=prev_path,
            )

            probs = pipeline.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)

            results = X.copy()
            results["default_probability"] = probs
            results["predicted_default"] = preds
            results["risk_label"] = results["predicted_default"].map({1: "High Risk", 0: "Low Risk"})

            st.success("Predictions generated successfully.")
            st.subheader("Prediction Preview")
            st.dataframe(results[["default_probability", "predicted_default", "risk_label"]].head(50), use_container_width=True)

            st.subheader("Risk Distribution")
            fig, ax = plt.subplots()
            results["risk_label"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("Risk Label")
            ax.set_ylabel("Count")
            ax.set_title("Predicted Risk Distribution")
            st.pyplot(fig)

            st.subheader("Download Predictions")
            csv = results[["default_probability", "predicted_default", "risk_label"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="loan_default_predictions.csv",
                mime="text/csv",
            )

    st.divider()
    st.subheader("What this project does")
    st.write(
        """
        - Merges customer demographics, performance history, and previous loans.
        - Aggregates multi-row loan history into customer-level features.
        - Trains a LightGBM classifier.
        - Evaluates with F1, Recall, PR-AUC, ROC-AUC, and confusion-matrix-based metrics.
        """
    )


if __name__ == "__main__":
    main()
