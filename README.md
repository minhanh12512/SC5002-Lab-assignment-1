# SC5002 Lab Assignment: Linear vs Ridge Regression

**Author:** Ho Thi Minh Anh 

**Course:** SC5002 Artificial Intelligence Fundamentals & Applications, Nanyang Technological University

**Semester:** 2026 Semester 1


## Project Overview

This repository contains a low-code machine learning notebook built on the California Housing dataset. It loads the data, standardises the features with `StandardScaler`, and compares Linear Regression with Ridge Regression at three penalty strengths (α = 0.1, 10, 1000). The models are compared using 5-fold cross-validation to see how L2 regularisation affects predictive accuracy.

## Repository Contents

| File | Description |
|---|---|
| `SC5002_ML_Notebook.ipynb` | Google Colab notebook with the code and outputs |
| `README.md` | Project documentation, setup and findings |

## Methodology & Setup

- **Data sampling:** 1,000 records are sampled from the dataset with `random_state=42`, so results are reproducible.
- **Data splitting:** 80/20 train/test split. 5-fold cross-validation is run on the training set only.
- **Feature scaling:** all input features are standardised with `StandardScaler` (mean 0, standard deviation 1).

## Requirements & How to Run

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `scikit-learn` (all pre-installed in Google Colab).

1. Open `SC5002_ML_Notebook.ipynb` in Google Colab (or click the badge above).
2. Select **Runtime > Run all**.

> **Note:** the notebook downloads the California Housing dataset through scikit-learn on the first run, so an internet connection is needed. No manual download is required.

## Results

| Model | Regularisation (α) | Mean 5-Fold R² |
|---|---|---|
| Linear Regression | None | 0.6457 |
| Ridge Regression | 0.1 | 0.6457 |
| Ridge Regression | 10.0 | 0.6443 |
| Ridge Regression | 1000.0 | 0.3574 |

![Model comparison chart](chart.png)

## Findings

- **α = 0.1 and α = 10 are effectively tied with Linear Regression.** Ridge can help when features are correlated (for example AveRooms and AveBedrms), but it gave no measurable benefit on this data. Since the scores are tied, the simpler model, Linear Regression, is the better choice.
- **α = 1000 underfits.** The penalty shrinks every feature weight strongly towards zero, so the model can only use a weak version of each feature. R² drops to 0.3574, losing almost half of its predictive power.

## Bug Fix

Running the original template raised `SyntaxError: unterminated f-string literal`. One `print` statement in the Ridge loop had been split across two lines, and an f-string must open and close on the same line.

```python
# Broken (split across two lines):
print(f"Ridge Regression (alpha={a:6.1f})
  Mean 5-Fold R2 Score: {scores.mean():.4f}")

# Fixed (one line):
print(f"Ridge Regression (alpha={a:6.1f}) Mean 5-Fold R2 Score: {scores.mean():.4f}")
```

## Limitations

- The 20% test set (`X_test`) is held out but never evaluated in the template; all scores come from cross-validation on the training set.
- `StandardScaler` is fitted on the full 1,000-row sample before the train/test split (`X_scaled = scaler.fit_transform(X)`), which leaks a small amount of information from the test set into training. Fitting the scaler inside a `Pipeline` would avoid this.

