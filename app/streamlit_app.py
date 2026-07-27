# app/streamlit_app.py

from __future__ import annotations

from pathlib import Path

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
        st.subheader("Upload the three CSV files")
        demo_file = st.file_uploader("traindemographics.csv", type=["csv"])
        perf_file = st.file_uploader("trainperf.csv", type=["csv"])
        prev_file = st.file_uploader("trainprevloans.csv", type=["csv"])

        run_btn = st.button("Generate Predictions", type="primary")

    with col2:
        st.subheader("Model Summary")
        st.write(f"**Threshold:** {threshold:.4f}")
        st.write(f"**Feature count:** {bundle.get('feature_count', 'N/A')}")
        st.write("**Target:** good_bad_flag")
        if "test_metrics" in bundle:
            st.json(bundle["test_metrics"])

    if run_btn:
        if not (demo_file and perf_file and prev_file):
            st.error("Please upload all three CSV files first.")
            st.stop()

        demographics = pd.read_csv(demo_file)
        perf = pd.read_csv(perf_file)
        prevloans = pd.read_csv(prev_file)

        X, y = prepare_inference_data(
            demographics_path=demo_file,
            perf_path=perf_file,
            prevloans_path=prev_file,
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
