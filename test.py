# test.py

import unittest
import numpy as np
import pandas as pd

from src.features import build_prevloan_features, build_model_frame
from src.utils import best_threshold_from_probs, compute_metrics
from src.train import build_pipeline


class TestLoanDefaultProject(unittest.TestCase):

    def test_prevloan_feature_aggregation(self):
        prevloans = pd.DataFrame(
            {
                "customerid": [1, 1, 2, 2, 2],
                "loannumber": [1, 2, 1, 2, 3],
                "loanamount": [100, 150, 200, 220, 250],
                "duedate": ["2024-01-01", "2024-02-01", "2024-01-10", "2024-02-10", "2024-03-10"],
                "closeddate": ["2024-01-05", "2024-02-20", "2024-01-09", "2024-02-15", "2024-03-20"],
            }
        )
        feat = build_prevloan_features(prevloans)
        self.assertIn("prevloan_count", feat.columns)
        self.assertIn("late_repayment_count", feat.columns)
        self.assertEqual(len(feat), 2)

    def test_model_frame_creation(self):
        demographics = pd.DataFrame(
            {
                "customerid": [1, 2],
                "birthdate": ["1990-01-01", "1988-06-15"],
                "bank_account_type": ["Savings", "Current"],
                "employment_status_clients": ["Self-Employed", "Employed"],
            }
        )
        perf = pd.DataFrame(
            {
                "customerid": [1, 2],
                "good_bad_flag": [0, 1],
                "loanamount": [1000, 1500],
                "totaldue": [1100, 1700],
                "approveddate": ["2024-01-01", "2024-01-02"],
                "creationdate": ["2023-12-28", "2024-01-01"],
            }
        )
        prevloans = pd.DataFrame(
            {
                "customerid": [1, 1, 2],
                "loannumber": [1, 2, 1],
                "loanamount": [100, 120, 200],
                "duedate": ["2024-01-01", "2024-02-01", "2024-01-10"],
                "closeddate": ["2024-01-05", "2024-02-20", "2024-01-09"],
            }
        )

        X, y = build_model_frame(demographics, perf, prevloans)
        self.assertEqual(len(X), 2)
        self.assertEqual(len(y), 2)
        self.assertTrue("good_bad_flag" not in X.columns)

    def test_threshold_and_metrics(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, f1 = best_threshold_from_probs(y_true, y_prob)
        self.assertTrue(0.0 <= threshold <= 1.0)
        self.assertTrue(0.0 <= f1 <= 1.0)

        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        self.assertIn("f1", metrics)
        self.assertIn("pr_auc", metrics)

    def test_pipeline_fits(self):
        X = pd.DataFrame(
            {
                "loanamount": [100, 120, 130, 140, 150, 160],
                "totaldue": [110, 130, 140, 160, 170, 180],
                "bank_account_type": ["Savings", "Current", "Savings", "Savings", "Current", "Current"],
                "employment_status_clients": ["Employed", "Self-Employed", "Employed", "Employed", "Self-Employed", "Employed"],
            }
        )
        y = pd.Series([0, 0, 0, 1, 1, 1])

        scale_pos_weight = (len(y) - y.sum()) / max(int(y.sum()), 1)
        pipe = build_pipeline(X, scale_pos_weight=scale_pos_weight)
        pipe.fit(X, y)
        probs = pipe.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), len(X))


if __name__ == "__main__":
    unittest.main()
