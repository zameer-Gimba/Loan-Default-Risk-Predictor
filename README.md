# Loan Default Risk Predictor

A machine learning project that predicts loan default risk using the Data Science Nigeria (DSN) Challenge 1 dataset from Zindi.

## Dataset
- `traindemographics.csv`
- `trainperf.csv`
- `trainprevloans.csv`

## Model
- LightGBM Classifier
- 5-fold Stratified Cross-Validation
- Holdout test split: 80/20

## Metrics
- F1-Score
- Recall
- PR-AUC
- ROC-AUC
- Precision
- Balanced Accuracy

## Project Structure
- `src/` — data prep, feature engineering, training, prediction
- `app/` — Streamlit app
- `models/` — saved model artifact
- `test.py` — unit tests

## Run Training
```bash
python -m src.train
```
## Run App
```
streamlit run app/streamlit_app.py
```
## Run Tests
```
python test.py
```
## Run order

```bash
python -m src.train
streamlit run app/streamlit_app.py
python test.py
```
