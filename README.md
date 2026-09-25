# Telco Customer Churn Prediction

A leakage-safe, end-to-end binary classification project that predicts which telecom customers are likely to churn, built to practice — and demonstrate — the habits that separate a real ML workflow from a tutorial notebook: proper train/test hygiene, honest evaluation under class imbalance, and measuring whether design decisions (feature engineering, tuning, ensembling) actually help instead of assuming they do.

## Problem

Telecom companies lose revenue every time a customer churns, and it's far cheaper to retain an existing customer than acquire a new one. The goal here is to flag customers who are likely to leave *before* they do, so a retention team can intervene — which makes **recall** (catching actual churners) and **PR-AUC** more important business metrics than plain accuracy on this dataset, where ~26% of customers churned.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 original features (demographics, account info, subscribed services), binary `Churn` target.

## Approach

```
Raw data → cleaning → feature engineering → train/test split (stratified) →
leakage-safe preprocessing pipeline → baseline models → 5-fold stratified CV →
hyperparameter tuning → final test-set evaluation → interpretation → save pipeline
```

**Key engineering decisions:**
- All preprocessing (imputation, scaling, one-hot encoding) is wrapped in scikit-learn `Pipeline` / `ColumnTransformer` objects and fit **only on training data** — no statistic used to transform a feature is ever computed on data the model will later be tested on.
- `TotalCharges` blanks were traced to zero-tenure customers and imputed with `tenure × MonthlyCharges` (a documented, domain-grounded rule) rather than a generic mean/median fill.
- Nominal categoricals (e.g., `Contract`, `PaymentMethod`) are one-hot encoded rather than label-encoded, to avoid imposing a false numeric order.
- Class imbalance is handled via `class_weight="balanced"` rather than naively optimizing accuracy.
- The engineered features (`total_addon_services`, `is_month_to_month`, `is_new_customer`, `avg_monthly_charge_per_tenure`) were **A/B tested** against a raw-feature-only baseline using the identical model and preprocessing recipe, rather than assumed to help.
- Every model was tuned with `GridSearchCV`/`RandomizedSearchCV` using 5-fold **stratified** cross-validation on the training set only. The test set was touched exactly once, at the end, for final reporting — never used to select a model or a hyperparameter.
- As a follow-up experiment, a **Bagging ensemble of 25 tuned Logistic Regression models** was tested against the single tuned model, to check whether ensembling helps a low-variance linear model (see Results).

## Models compared

Logistic Regression · Decision Tree · Random Forest · SVM (RBF) · Gradient Boosting — each with a tuned and untuned (baseline) version, plus a bagged Logistic Regression ensemble.

## Results

**Baseline vs. tuned (test set):**

| Model | Baseline F1 | Tuned F1 | Baseline PR-AUC | Tuned PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.617 | 0.622 | 0.662 | 0.657 |
| Decision Tree | 0.492 | 0.621 | 0.379 | 0.547 |
| Random Forest | 0.530 | 0.639 | 0.608 | 0.657 |
| SVM | 0.633 | 0.629 | 0.584 | 0.612 |

Tuning produced its biggest gains on Decision Tree (+0.129 F1) and Random Forest (+0.109 F1) — both high-variance model families where unconstrained depth was overfitting in the baseline. Tuning had little effect on SVM/Logistic Regression, which were already closer to their achievable optimum at default settings.

**Final model selection (5-fold CV on training data, F1-ranked):**

| Model | CV F1 | CV ROC-AUC | CV PR-AUC |
|---|---|---|---|
| **Tuned Random Forest** | **0.640** | 0.849 | 0.665 |
| Tuned Logistic Regression | 0.634 | 0.849 | 0.667 |
| Tuned SVM | 0.625 | 0.833 | 0.623 |
| Tuned Decision Tree | 0.619 | 0.817 | 0.562 |
| Gradient Boosting (baseline) | 0.584 | 0.846 | 0.662 |

**Final model: Tuned Random Forest — held-out test set performance:**

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.766 | 0.541 | 0.781 | 0.639 | 0.844 | 0.657 |

The final model correctly identifies **~78% of customers who actually churn** (recall), at a precision of ~54% — a deliberate trade-off favoring catching more at-risk customers over avoiding false alarms, appropriate given that a missed churner is typically costlier than an unnecessary retention offer.

**Ensembling follow-up — does bagging help Logistic Regression?**

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Single Tuned Logistic Regression | 0.622 | 0.846 | 0.657 |
| Bagged Logistic Regression (25 models) | 0.626 | 0.846 | 0.659 |

Bagging produced only a marginal, likely noise-level improvement — consistent with the underlying theory: Logistic Regression is a low-variance estimator (it converges to essentially the same solution regardless of which bootstrap sample it sees), so there's little variance for bagging to average away. This is included as a deliberate negative result, not omitted — an honest ablation is more informative than a cherry-picked one.

## What I learned

- Why `Pipeline`/`ColumnTransformer` exist and what specifically breaks without them (data leakage from fitting preprocessing on the full dataset before splitting).
- Why accuracy is a misleading headline metric under class imbalance, and why PR-AUC is more informative than ROC-AUC when the positive class is rare.
- How to test whether a design decision (a feature, a tuning pass, an ensemble) actually helped, instead of assuming it did — and how to report a null result honestly.
- The mechanics and tuning intuition behind five model families: the sigmoid/log-loss objective and regularization (`C`) in Logistic Regression, Gini/entropy splitting in Decision Trees, bagging + random feature subsets in Random Forest, margin maximization and the RBF kernel in SVM, and sequential residual-fitting in Gradient Boosting.
- Why bagging helps high-variance models (trees) far more than low-variance ones (linear models) — verified empirically, not just asserted.
- Why hyperparameter selection must happen via cross-validation on the training set alone, with the test set reserved for a single final evaluation.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib

## Repo structure

```
├── churn_prediction_complete_with_LR_ensemble.ipynb   # full analysis + modeling
├── telco_churn_final_pipeline.joblib                  # saved final pipeline (preprocessing + model)
└── README.md
```

## How to run

```bash
pip install -r requirements.txt   # pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, joblib
jupyter notebook churn_prediction_complete_with_LR_ensemble.ipynb
```

## Limitations

This dataset is observational, not experimental — associations found in EDA (e.g., month-to-month contracts correlating with higher churn) describe patterns in the data, not proven causal effects of an intervention. The model estimates churn *risk*; deciding what action to take on that risk (and at what probability threshold) is a separate business decision, explored briefly in the notebook's threshold-tuning section.s