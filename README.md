# Telco Customer Churn — Prediction Pipeline

End-to-end ML pipeline on the Telco Customer Churn dataset: EDA, preprocessing, feature encoding, model training, and evaluation.

## Dataset

[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customer records, 21 columns (demographics, account info, services subscribed, and churn label).

Download `Churn_Data.csv` from the link above and place it in the repo root before running the notebook.

## Contents

- `churn_prediction_pipeline.ipynb` — data loading, cleaning, EDA, multivariate analysis, feature encoding, model training, and evaluation (built up in stages as the project progresses).
- `requirements.txt` — Python dependencies.

## Key Findings

- Customers with **low tenure on month-to-month contracts** churn the most.
- Customers retained through at least a **one-year contract** are far more likely to stay.
- **Fiber optic** internet customers churn more than **DSL** customers.
- Customers **without TechSupport or OnlineSecurity** tend to churn more.
- **Electronic check** payment users churn significantly more than automatic payment users (bank transfer / credit card).

**Emerging high-risk profile:** month-to-month contract + short tenure + fiber optic internet + no TechSupport/OnlineSecurity + electronic check payment.

## Next Steps

- Feature engineering & encoding (categorical variables, handling the "No internet service" redundancy across service columns)
- Train/test split (stratified, due to class imbalance)
- Baseline model (Logistic Regression / Decision Tree)
- Evaluation with precision/recall/F1 (accuracy alone is misleading on imbalanced data)

## Setup

```bash
pip install -r requirements.txt
```

Then open `churn_eda.ipynb` in Jupyter or Google Colab.
