

# Credit Risk Prediction —  ML System

## Overview

This project implements an end-to-end machine learning system for predicting loan default risk using the Home Credit Default Risk dataset.

The goal is to build an  ML pipeline that:

* Handles real-world messy tabular financial data
* Uses modular feature engineering and preprocessing pipelines
* Benchmarks multiple model families
* Selects a final production candidate based on validation performance
* Provides interpretability through feature importance analysis

This project focuses on **engineering discipline + reproducibility**, not just model accuracy.

---

## Business Problem

Financial institutions must estimate the probability that a customer will default on a loan.

Accurate credit risk prediction enables:

* Better loan approval decisions
* Risk-adjusted interest rate pricing
* Reduced default losses
* Improved portfolio stability

This system predicts:

**Input:** Customer application data
**Output:** Probability of default (PD)

---

## Dataset

Source: Home Credit Default Risk Dataset

Main Table Used:

* `application_train.csv`

Future extensions could include:

* Bureau history
* Previous loans
* Installment payment history

---

## Project Architecture

```
Raw Data
↓
Feature Engineering Module (src/features)
↓
Preprocessing Pipeline (ColumnTransformer)
↓
Model Training Modules (src/models)
↓
Model Evaluation + Cross Validation
↓
Final Model Selection
↓
Model Serialization (Inference Ready)
```

---

## Repository Structure

```
src/
  features/
    build_features.py
    pipeline.py

  models/
    train_model.py
    train_tree_model.py
    train_histgb_model.py
    train_xgb_model.py
    evaluate_model.py
    save_model.py

notebooks/
  01_eda.ipynb
  03_modeling.ipynb

artifacts/
  saved_model.joblib
```

---

## Feature Engineering

Key engineered features include:

### Financial Stress Indicators

* Credit to Income Ratio
* Annuity to Income Ratio

### Stability Indicators

* Employment anomaly flags
* Registration duration signals
* Phone activity recency

### Demographic Signals

* Age (converted from DAYS_BIRTH)

### Data Cleaning

* Sentinel missing value handling
* Identifier column removal
* Redundant feature removal

---

## Preprocessing Pipeline

Implemented using sklearn `ColumnTransformer`.

### Numeric Pipeline

* Median Imputation
* Standard Scaling

### Categorical Pipeline

* Most Frequent Imputation
* One-Hot Encoding (unknown category safe)

---

## Models Evaluated

### 1️. Logistic Regression

Baseline linear model.

Purpose:

* Establish linear separability baseline
* Provide interpretable reference performance

---

### 2️. Random Forest

Tree ensemble baseline.

Purpose:

* Capture nonlinear interactions
* Provide feature importance baseline

---

### 3️. Gradient Boosting (Sklearn)

Sequential boosting baseline.

Purpose:

* Test boosting improvement over bagging

---

### 4️. HistGradientBoosting

Modern histogram boosting (sklearn optimized).

Purpose:

* Faster boosting for large tabular datasets
* Improved generalization vs classic GB

---

### 5️. XGBoost (Final Selected Model)

Industry-standard gradient boosting.

Purpose:

* Final performance optimization
* Strong tabular interaction learning
* Robust regularization controls

---

## Model Performance

### Validation ROC AUC

| Model                | ROC AUC |
| -------------------- | ------- |
| Logistic Regression  | ~0.749  |
| Random Forest        | ~0.726  |
| Gradient Boosting    | ~0.753  |
| HistGradientBoosting | ~0.759  |
| XGBoost              | ~0.762  |

---

###  ROC Curve Comparison

![ROC Curve Comparison](assets/roc_curve.png)

---

###  Cross Validation Stability

Mean CV ROC AUC (Logistic Baseline): ~0.746
Std Dev: ~0.0026

Indicates stable generalization across folds.

---

## Final Model Selection

Final Production Model: **XGBoost**

Selected because:

* Highest validation ROC AUC
* Consistent improvement across model ladder
* Strong handling of tabular feature interactions
* Industry standard for tabular ML

---

## Model Interpretability

Feature importance analysis shows dominant signals from:

* External credit risk score features (EXT_SOURCE)
* Payment burden ratios
* Customer stability indicators
* Age / lifecycle effects

---

###  Feature Importance Plot

![Top Feature Importance](assets/top_feat_imp.png)

---

## Model Deployment Readiness

Final model is saved as serialized pipeline artifact:

```
artifacts/xgb_credit_model.joblib
```

Includes:
Preprocessing
Feature engineering
Model inference

Allows direct prediction on raw input data.

---

## How to Run

### Install Dependencies

```
pip install -r requirements.txt
```

---

### Train Models

Run modeling notebook:

```
notebooks/03_modeling.ipynb
```

---

### Load Saved Model

```python
import joblib

model = joblib.load("artifacts/xgb_credit_model.joblib")
preds = model.predict_proba(X_new)
```

---

## Key Technical Learnings

* Feature engineering dominates tabular ML performance
* Boosting models outperform bagging on structured financial data
* Histogram boosting significantly improves training efficiency
* Cross validation is essential for stable performance estimation
* Modular pipeline design is critical for production ML systems

---

## Future Improvements

Potential next steps:

* Multi-table feature aggregation (bureau, previous loans)
* Probability calibration for risk pricing
* Decision threshold optimization for profit maximization
* Model monitoring and drift detection simulation

---

## Author

Built as an end-to-end production-style ML system focusing on:
Reproducibility
Modularity
Real-world tabular ML workflow

---

