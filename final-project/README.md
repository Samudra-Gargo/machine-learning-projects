# Predictive Maintenance of Industrial Equipment Using Machine Learning

## Overview

This project develops and evaluates machine learning models for predicting
industrial machine failures using the **AI4I 2020 Predictive Maintenance
Dataset** from the UCI Machine Learning Repository.

The primary objective is to determine whether a machine is likely to fail
based on operational and sensor measurements. Since machine failures are
rare compared with normal operation, the project focuses on metrics such as
precision, recall, F1-score, ROC-AUC, and confusion matrices rather than
accuracy alone.

A key part of the project is **decision-threshold tuning**, investigating
thresholds from 0.1 to 0.9 to understand the trade-off between detecting more
failures and generating false alarms.

---

## Dataset

The AI4I 2020 Predictive Maintenance Dataset contains **10,000 machine
operation records**.

### Input Features

- **Type** — categorical machine/product type
- **Air temperature [K]** — surrounding air temperature
- **Process temperature [K]** — machine/process operating temperature
- **Rotational speed [rpm]** — shaft or motor speed
- **Torque [Nm]** — mechanical load on the machine
- **Tool wear [min]** — accumulated tool wear time

### Target

**Machine failure**

- `0` — No machine failure
- `1` — Machine failure

The dataset is highly imbalanced:

- Normal operation: **9,661 samples**
- Machine failures: **339 samples**
- Failure rate: **3.39%**

Because of this imbalance, a model can achieve high accuracy while still
performing poorly at detecting actual failures.

---

## Methodology

### 1. Data Preprocessing

The categorical `Type` feature was converted into numerical features using
one-hot encoding.

The dataset was divided into:

- **Training set** — model training
- **Validation set** — hyperparameter tuning
- **Test set** — final evaluation

The held-out test set contained **2,000 samples**, including 68 failure
cases.

Feature scaling was performed using `StandardScaler` for models sensitive
to feature magnitude. Scaling was fitted only on the training data to
prevent data leakage.

---

## Machine Learning Models

The following models were evaluated:

### Logistic Regression

Used as a baseline linear classification model.

- `class_weight="balanced"`
- Regularization parameter `C` was tuned
- Best validation configuration occurred around `C = 10`

### Decision Tree

Used to capture nonlinear relationships between machine operating
conditions and failures.

- Tuned parameter: `max_depth`
- Best depth: `14`

### Random Forest

An ensemble of decision trees used to improve robustness and reduce the
overfitting risk of a single decision tree.

Key parameters included:

- `n_estimators`
- `max_depth`

### Scikit-learn MLP

A neural-network baseline implemented using Scikit-learn's
`MLPClassifier`.

Different hidden-layer configurations, regularization values, and learning
rates were evaluated.

### Custom PyTorch MLP

A neural network implemented directly in PyTorch.

The custom implementation provided control over:

- Forward propagation
- Loss calculation
- Backpropagation
- Optimizer updates
- Training loss tracking
- Validation loss tracking

Binary classification was performed using `BCEWithLogitsLoss`.

### XGBoost

XGBoost was evaluated as a boosted-tree model for structured tabular data.

The tuned parameters included:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

Best validation configuration was approximately:

```text
n_estimators = 200
max_depth = 5
learning_rate = 0.2
