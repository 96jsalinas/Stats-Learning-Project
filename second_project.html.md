---
title: "Credit Card Default: Machine Learning Tools"
format:
    html:
        code-fold: true
        toc: true
        toc-depth: 3
        embed-resources: true
        keep-md: true
jupyter: python3
execute:
    cache: true
authors:
  - name: Josu Salinas Colina
  - name: Eoin Gallagher
  - name: Francisca Eeckels
---

# Dependencies

::: {#9c4bc62a .cell execution_count=1}
``` {.python .cell-code}
import subprocess, sys

_pkgs = ["xgboost", "shap", "torch"]
for _pkg in _pkgs:
    try:
        __import__(_pkg)
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", _pkg, "-q"], check=True)
```
:::


# Overview

This report is the second part of our analysis of the **UCI Default of Credit Card Clients** dataset (30 000 Taiwanese cardholders, 2005). The goal remains predicting whether a client will default on their October 2005 payment.

Part 1 covered exploratory data analysis, a structured preprocessing pipeline, and three generative / probabilistic classifiers (LDA, QDA, Gaussian Naïve Bayes) together with k-Nearest Neighbours and a detailed cost-sensitive learning framework. The key benchmark established there was an **expected cost of 0.599** using LDA probabilities at a threshold of 0.20 (cost matrix: FN = 5, FP = 1). This threshold was the grid-search optimum from Part 1, Section 7; the analytical cost-optimal threshold derived from the cost ratio is p* = FP/(FP + FN) = 1/6 ≈ 0.167.

This report applies the remaining families of machine learning classifiers from the course:

| Section | Methods |
|---|---|
| **2. Support Vector Machines** | Linear SVM, RBF kernel SVM (C × γ grid search), cost-sensitive SVM |
| **3. Tree-Based Methods** | Decision tree + pruning, Bagging, Random Forest, GBM, XGBoost |
| **4. Neural Networks** | MLP (PyTorch), hyperparameter search, SHAP explainability |
| **5. Overall Comparison** | All models from Parts 1 and 2 |

k-Nearest Neighbours is not repeated here as it was covered in Part 1, Section 8.

---

# 1. Setup: Data & Preprocessing

The pipeline below is identical to Part 1. It loads the dataset, cleans and encodes categorical variables, applies a selective log-transform to skewed monetary columns, performs a stratified 70/30 train–test split, and standardises features using a scaler fitted exclusively on the training set. The Part 1 reference models (LDA, GNB, k-NN) are also re-fitted here so their predictions are available for the final comparison in Section 5.

::: {#ac9d73d2 .cell execution_count=2}
``` {.python .cell-code}
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

# ── Load ──────────────────────────────────────────────────────────────────────
local_file = "default_credit_card_clients.xlsx"
url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
       "/00350/default%20of%20credit%20card%20clients.xls")

if os.path.exists(local_file):
    df = pd.read_excel(local_file)
else:
    df = pd.read_excel(url, header=1)
    df.to_excel(local_file, index=False)

rename_dict = {
    'LIMIT_BAL': 'credit_limit', 'SEX': 'gender',
    'EDUCATION': 'education', 'MARRIAGE': 'marital_status', 'AGE': 'age',
    'PAY_0': 'status_sep', 'PAY_2': 'status_aug', 'PAY_3': 'status_jul',
    'PAY_4': 'status_jun', 'PAY_5': 'status_may', 'PAY_6': 'status_apr',
    'BILL_AMT1': 'bill_sep', 'BILL_AMT2': 'bill_aug', 'BILL_AMT3': 'bill_jul',
    'BILL_AMT4': 'bill_jun', 'BILL_AMT5': 'bill_may', 'BILL_AMT6': 'bill_apr',
    'PAY_AMT1': 'paid_sep', 'PAY_AMT2': 'paid_aug', 'PAY_AMT3': 'paid_jul',
    'PAY_AMT4': 'paid_jun', 'PAY_AMT5': 'paid_may', 'PAY_AMT6': 'paid_apr',
    'default payment next month': 'default'
}
df.rename(columns=rename_dict, inplace=True)
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# ── Clean & encode ────────────────────────────────────────────────────────────
df['education']      = df['education'].replace([0, 5, 6], 4)
df['marital_status'] = df['marital_status'].replace(0, 3)

nominal_cols = ['gender', 'marital_status']
df_encoded   = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

X = df_encoded.drop(columns=['default'], errors='ignore')
y = df_encoded['default']

# ── Train / test split (stratified 70/30) ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ── Selective log-transform (|skew| > 0.75, training set only) ───────────────
monetary_cols = [
    'credit_limit', 'age',
    'bill_sep', 'bill_aug', 'bill_jul', 'bill_jun', 'bill_may', 'bill_apr',
    'paid_sep', 'paid_aug', 'paid_jul', 'paid_jun', 'paid_may', 'paid_apr'
]
ordinal_cols  = [
    'education',
    'status_sep', 'status_aug', 'status_jul',
    'status_jun', 'status_may', 'status_apr'
]

SKEW_THRESHOLD = 0.75
for col in monetary_cols:
    if abs(X_train[col].skew()) > SKEW_THRESHOLD:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col]  = np.log1p(X_test[col].clip(lower=0))

# ── Standardise ───────────────────────────────────────────────────────────────
final_cols = (monetary_cols + ordinal_cols +
              [c for c in X.columns if 'gender_' in c or 'marital_status_' in c])

X_train = X_train[final_cols]
X_test  = X_test[final_cols]

scaler        = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

X_train_scaled = X_train_scaled.reset_index(drop=True)
X_test_scaled  = X_test_scaled.reset_index(drop=True)
y_train        = y_train.reset_index(drop=True)
y_test         = y_test.reset_index(drop=True)

print(f"Training set : {X_train_scaled.shape[0]} rows | "
      f"default rate = {y_train.mean():.3f}")
print(f"Test set     : {X_test_scaled.shape[0]} rows  | "
      f"default rate = {y_test.mean():.3f}")
print(f"Feature count: {X_train_scaled.shape[1]}")
```

::: {.cell-output .cell-output-stdout}
```
Training set : 21000 rows | default rate = 0.221
Test set     : 9000 rows  | default rate = 0.221
Feature count: 24
```
:::
:::


::: {#253e7d5a .cell execution_count=3}
``` {.python .cell-code}
# ── Cost framework (carried over from Part 1) ─────────────────────────────────
C_cost = np.array([[0, 1],   # TN, FP
                   [5, 0]])  # FN, TP

opt_t = 0.200   # cost-optimal threshold (grid optimum from Part 1, Section 7)

def expected_cost(y_true, y_pred, cost_matrix=C_cost, normalize=True):
    cm    = confusion_matrix(y_true, y_pred)
    total = np.sum(cm * cost_matrix)
    return total / len(y_true) if normalize else total

def eval_model(name, y_pred, y_prob, threshold=None):
    """Return a metrics dict for a single model configuration."""
    label = name if threshold is None else f"{name} (t={threshold})"
    return {
        'Model':                  label,
        'Accuracy':               accuracy_score(y_test, y_pred),
        'Recall (Default)':       recall_score(y_test, y_pred),
        'Precision (Default)':    precision_score(y_test, y_pred, zero_division=0),
        'F1 (Default)':           f1_score(y_test, y_pred),
        'ROC-AUC':                roc_auc_score(y_test, y_prob),
        'Expected Cost':          expected_cost(y_test, y_pred),
    }

# Accumulator for the final comparison (Section 5)
all_results = []
all_probs   = {}   # name → y_prob array (for ROC overlay)
```
:::


::: {#e26bdf5c .cell execution_count=4}
``` {.python .cell-code}
# ── Part 1 reference models ────────────────────────────────────────────────────
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# LDA (empirical priors) — best ranking model from Part 1
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_prob_lda     = lda.predict_proba(X_test_scaled)[:, 1]
y_pred_lda_50  = lda.predict(X_test_scaled)
y_pred_lda_opt = (y_prob_lda >= opt_t).astype(int)

# GNB — best minority-class F1 among Part 1 probabilistic models
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_prob_gnb = gnb.predict_proba(X_test_scaled)[:, 1]
y_pred_gnb = gnb.predict(X_test_scaled)

# k-NN (k=59) — results carried over directly from Part 1 Section 8
# (refitting is omitted here: brute-force kNN prediction on 21k×9k is slow
#  and the result is deterministic given the same preprocessed data)
from sklearn.neighbors import KNeighborsClassifier
knn59 = KNeighborsClassifier(n_neighbors=59, algorithm='kd_tree', n_jobs=-1)
knn59.fit(X_train_scaled, y_train)
y_prob_knn = knn59.predict_proba(X_test_scaled)[:, 1]
y_pred_knn = knn59.predict(X_test_scaled)

# Register for final comparison
all_results += [
    eval_model('LDA', y_pred_lda_50,  y_prob_lda),
    eval_model('LDA', y_pred_lda_opt, y_prob_lda, threshold=opt_t),
    eval_model('GNB', y_pred_gnb,     y_prob_gnb),
    eval_model('k-NN (k=59)', y_pred_knn, y_prob_knn),
]
all_probs['LDA']         = y_prob_lda
all_probs['GNB']         = y_prob_gnb
all_probs['k-NN (k=59)'] = y_prob_knn

print("Part 1 reference models fitted.")
```

::: {.cell-output .cell-output-stdout}
```
Part 1 reference models fitted.
```
:::
:::


---

# 2. Support Vector Machines

## 2.1 Theory

Support Vector Machines (SVMs) find the decision boundary that **maximises the margin** between classes. The progression from the basic formulation to the full non-linear classifier follows three steps:

**Maximal Margin Classifier (hard margin):** assumes perfect linear separability and finds the hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ that maximises $\frac{2}{\|\mathbf{w}\|}$. Because real data are never perfectly separable this is mainly of theoretical interest.

**Support Vector Classifier (soft margin):** introduces slack variables $\xi_i \geq 0$ and a penalty parameter $C$: a large $C$ forces a tight margin with few violations (approaching the hard-margin case), while a small $C$ allows more violations in exchange for a wider, more robust margin.

**Support Vector Machine (kernel trick):** replaces the inner product $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$, implicitly mapping the data to a higher-dimensional feature space where a linear boundary becomes non-linear in the original space. The **Radial Basis Function (RBF)** kernel is:

$$K(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\gamma\,\|\mathbf{x} - \mathbf{x}'\|^2\right)$$

$\gamma$ controls the reach of each support vector: large $\gamma$ means each point influences only its immediate neighbourhood (complex, potentially overfit boundary); small $\gamma$ gives a smoother, more global boundary.

---

## 2.2 Linear SVM (Baseline)

### Fit & Evaluate

::: {#91811e0e .cell execution_count=5}
``` {.python .cell-code}
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV

# LinearSVC is 10-100× faster than SVC(kernel='linear') on large datasets.
# CalibratedClassifierCV adds Platt-scaling in a separate CV step so we still
# get calibrated probabilities for ROC-AUC and threshold tuning.
_svm_lin_base = LinearSVC(C=1.0, max_iter=5000, random_state=42)
svm_lin = CalibratedClassifierCV(_svm_lin_base, cv=3)
svm_lin.fit(X_train_scaled, y_train)

y_pred_svm_lin     = svm_lin.predict(X_test_scaled)
y_prob_svm_lin     = svm_lin.predict_proba(X_test_scaled)[:, 1]
y_pred_svm_lin_opt = (y_prob_svm_lin >= opt_t).astype(int)
auc_svm_lin        = roc_auc_score(y_test, y_prob_svm_lin)

print("=== Linear SVM (C=1): Classification Report ===")
print(classification_report(y_test, y_pred_svm_lin,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_svm_lin:.4f}")
print(f"Accuracy      (t=0.50): {accuracy_score(y_test, y_pred_svm_lin):.4f}")
print(f"Recall Default(t=0.50): {recall_score(y_test, y_pred_svm_lin):.4f}")
print(f"Recall Default(t={opt_t}): {recall_score(y_test, y_pred_svm_lin_opt):.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_svm_lin):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_svm_lin_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
=== Linear SVM (C=1): Classification Report ===
              precision    recall  f1-score   support

  No Default       0.82      0.96      0.88      7009
     Default       0.65      0.24      0.36      1991

    accuracy                           0.80      9000
   macro avg       0.74      0.60      0.62      9000
weighted avg       0.78      0.80      0.77      9000

ROC-AUC: 0.7394
Accuracy      (t=0.50): 0.8042
Recall Default(t=0.50): 0.2446
Recall Default(t=0.2): 0.6504
Expected cost (t=0.50): 0.8642
Expected cost (t=0.2): 0.6062
```
:::
:::


::: {#fb1899b8 .cell execution_count=6}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_svm_lin),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('Linear SVM — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_svm_lin, ax=axes[1],
                                 name=f'Linear SVM (AUC = {auc_svm_lin:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('Linear SVM — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-7-output-1.png){width=974 height=372}
:::
:::


### Interpretation

* The linear SVM achieves **80.4 % accuracy** and a **ROC-AUC of 0.7394**, nearly identical to Part 1's LDA (0.7401). This is expected: both methods produce a linear boundary, and LDA's discriminant function is close to the SVM's hyperplane when the class-conditional distributions are roughly Gaussian.
* Recall on the default class is only **24.5 %** at the standard threshold, meaning three in four defaulters are missed — the same majority-class bias observed with LDA in Part 1.
* Lowering the threshold to 0.20 raises recall to **65.0 %** and cuts expected cost from **0.8642 to 0.6062**, confirming that even a linear model benefits substantially from cost-aware thresholding.
* These results set a clean linear baseline; the next step is to test whether the RBF kernel can capture non-linear interactions that improve discrimination.

---

## 2.3 RBF Kernel SVM

### Theory

The linear SVM assumes the classes are approximately linearly separable in the standardised feature space. The bill-amount multicollinearity noted in Part 1 (Section 3.3) and the moderate class overlap suggest that a kernel mapping may reveal structure not accessible to a linear boundary. The RBF kernel is the natural first choice: it is universal (can approximate any continuous decision function) and is governed by two interpretable hyperparameters, $C$ and $\gamma$.

### Hyperparameter Tuning

To keep grid-search runtime manageable, we tune on a stratified 5 000-row subsample of the training set and refit the best configuration on the full 21 000-row training set.

::: {#1bcc1ae8 .cell execution_count=7}
``` {.python .cell-code}
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Subsample for grid search (SVC scales as O(n²–n³));
# 5 000 rows and a 3×3 grid keeps runtime to ~5 min on CPU.
rng = np.random.default_rng(42)
tune_idx = rng.choice(len(X_train_scaled), size=5000, replace=False)
X_tune   = X_train_scaled.iloc[tune_idx].values
y_tune   = y_train.iloc[tune_idx].values

param_grid = {
    'C':     [0.1, 10, 100],
    'gamma': [0.001, 0.01, 0.1],
}

cv_svm = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# probability=False during search (uses decision_function → faster, no Platt scaling)
svm_rbf_gs = GridSearchCV(
    SVC(kernel='rbf', probability=False, random_state=42),
    param_grid,
    cv=cv_svm,
    scoring='roc_auc',
    n_jobs=-1,
    refit=False,
)
svm_rbf_gs.fit(X_tune, y_tune)

best_params_rbf = svm_rbf_gs.best_params_
print(f"Best params (subsample CV): {best_params_rbf}")
print(f"Best CV AUC: {svm_rbf_gs.best_score_:.4f}")
```

::: {.cell-output .cell-output-stdout}
```
Best params (subsample CV): {'C': 10, 'gamma': 0.001}
Best CV AUC: 0.7390
```
:::
:::


::: {#8b236a77 .cell execution_count=8}
``` {.python .cell-code}
# Heatmap of CV AUC across the C × gamma grid
cv_results = pd.DataFrame(svm_rbf_gs.cv_results_)
pivot = cv_results.pivot_table(
    index='param_C', columns='param_gamma', values='mean_test_score'
)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
plt.title('RBF SVM: Grid-Search CV ROC-AUC (C × γ)')
plt.xlabel('gamma (γ)')
plt.ylabel('C')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-9-output-1.png){width=710 height=468}
:::
:::


### Fit & Evaluate

::: {#7a4c1b18 .cell execution_count=9}
``` {.python .cell-code}
# Refit best config on full training set with probability=True for calibrated scores
svm_rbf = SVC(kernel='rbf',
              C=best_params_rbf['C'],
              gamma=best_params_rbf['gamma'],
              probability=True,
              random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

y_pred_svm_rbf     = svm_rbf.predict(X_test_scaled)
y_prob_svm_rbf     = svm_rbf.predict_proba(X_test_scaled)[:, 1]
y_pred_svm_rbf_opt = (y_prob_svm_rbf >= opt_t).astype(int)
auc_svm_rbf        = roc_auc_score(y_test, y_prob_svm_rbf)

print(f"=== RBF SVM (C={best_params_rbf['C']}, γ={best_params_rbf['gamma']}): "
      f"Classification Report ===")
print(classification_report(y_test, y_pred_svm_rbf,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_svm_rbf:.4f}")
print(f"Accuracy      (t=0.50): {accuracy_score(y_test, y_pred_svm_rbf):.4f}")
print(f"Recall Default(t=0.50): {recall_score(y_test, y_pred_svm_rbf):.4f}")
print(f"Recall Default(t={opt_t}): {recall_score(y_test, y_pred_svm_rbf_opt):.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_svm_rbf):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_svm_rbf_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
=== RBF SVM (C=10, γ=0.001): Classification Report ===
              precision    recall  f1-score   support

  No Default       0.83      0.96      0.89      7009
     Default       0.66      0.29      0.40      1991

    accuracy                           0.81      9000
   macro avg       0.74      0.62      0.64      9000
weighted avg       0.79      0.81      0.78      9000

ROC-AUC: 0.7082
Accuracy      (t=0.50): 0.8094
Recall Default(t=0.50): 0.2853
Recall Default(t=0.2): 0.4380
Expected cost (t=0.50): 0.8230
Expected cost (t=0.2): 0.6943
```
:::
:::


::: {#27fe572a .cell execution_count=10}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_svm_rbf),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('RBF SVM — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_svm_rbf, ax=axes[1],
                                 name=f'RBF SVM (AUC = {auc_svm_rbf:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('RBF SVM — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-11-output-1.png){width=974 height=372}
:::
:::


### Interpretation

* The grid search selects **C = 10, γ = 0.001** with a subsample CV AUC of 0.7390. On the full test set the RBF SVM achieves **ROC-AUC = 0.7082**, which is notably **lower** than the linear SVM (0.7394). This suggests that non-linear structure in this dataset is limited: the RBF kernel introduces flexibility that the model cannot exploit productively, and the high-dimensional feature space (24 features) may cause the kernel mapping to produce a noisier boundary.
* Accuracy at the default threshold (80.9 %) and recall on the default class (28.5 %) are marginally better than the linear SVM's, but the drop in AUC means the model's overall ranking quality is worse.
* Applying the cost-optimal threshold 0.20 improves recall to **43.8 %** but expected cost only decreases to **0.6943** — considerably worse than the linear SVM at the same threshold (0.6062). The calibrated probabilities from the RBF SVM are less well-separated, making threshold tuning less effective.
* The heatmap reveals that CV AUC is relatively flat across the grid, with large γ values (γ ≥ 0.1) performing worst — consistent with over-localised decision boundaries that overfit to the training subsample.

---

## 2.4 Cost-Sensitive SVM

The cost matrix assigns a 5× penalty to false negatives. The most direct way to encode this in an SVM is via the `class_weight` parameter, which scales the $C$ penalty differently for each class — equivalently it moves the decision boundary toward the majority class.

::: {#f6b24082 .cell execution_count=11}
``` {.python .cell-code}
# class_weight matches the FN:FP cost ratio (5:1)
svm_rbf_cw = SVC(kernel='rbf',
                 C=best_params_rbf['C'],
                 gamma=best_params_rbf['gamma'],
                 class_weight={0: 1, 1: 5},
                 probability=True,
                 random_state=42)
svm_rbf_cw.fit(X_train_scaled, y_train)

y_pred_svm_cw     = svm_rbf_cw.predict(X_test_scaled)
y_prob_svm_cw     = svm_rbf_cw.predict_proba(X_test_scaled)[:, 1]
y_pred_svm_cw_opt = (y_prob_svm_cw >= opt_t).astype(int)
auc_svm_cw        = roc_auc_score(y_test, y_prob_svm_cw)

print("=== RBF SVM (class_weight={0:1, 1:5}): Classification Report ===")
print(classification_report(y_test, y_pred_svm_cw,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_svm_cw:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_svm_cw):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_svm_cw_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
=== RBF SVM (class_weight={0:1, 1:5}): Classification Report ===
              precision    recall  f1-score   support

  No Default       0.89      0.67      0.76      7009
     Default       0.38      0.72      0.50      1991

    accuracy                           0.68      9000
   macro avg       0.64      0.69      0.63      9000
weighted avg       0.78      0.68      0.70      9000

ROC-AUC: 0.7576
Expected cost (t=0.50): 0.5736
Expected cost (t=0.2): 0.5766
```
:::
:::


::: {#82c05300 .cell execution_count=12}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for ax, (title, yp) in zip(axes, [
    ('RBF SVM: default (t=0.50)',       y_pred_svm_rbf),
    (f'RBF SVM: cost threshold (t={opt_t})', y_pred_svm_rbf_opt),
    ('RBF SVM: class_weight {0:1,1:5}', y_pred_svm_cw),
]):
    ConfusionMatrixDisplay(confusion_matrix(y_test, yp),
                           display_labels=['No Default', 'Default']).plot(
        ax=ax, colorbar=False)
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-13-output-1.png){width=1520 height=372}
:::
:::


### Interpretation

* Encoding the 5:1 cost ratio directly via `class_weight={0:1, 1:5}` is the most effective SVM strategy. It raises recall on defaulters to **71.7 %** — roughly 2.5 times the standard RBF SVM's 28.5 % — and achieves the **lowest expected cost among all SVM variants (0.5736)**.
* The trade-off is a significant drop in accuracy (from 81 % to **67.7 %**) and precision on the default class (from 66 % to **37.9 %**). This is the classic cost-sensitive effect: the model now predicts default more aggressively, correctly catching more true defaulters but also flagging more false positives.
* Interestingly, the class-weighted SVM also produces the **highest ROC-AUC (0.7576)** of all three SVM variants. The class weighting changes the objective during SVM training, not just the threshold — the support vectors are selected differently, leading to a genuinely better ranking model.
* Comparing the confusion matrices side-by-side: at t = 0.50, the default RBF SVM predicts "default" for very few cases (high precision, low recall), while the class-weighted variant flips this trade-off. The cost-threshold variant at t = 0.20 falls between the two.
* The expected costs at t = 0.50 and t = 0.20 for the class-weighted SVM are nearly identical (**0.5736 vs 0.5766**), suggesting that the learned boundary already internalises the cost asymmetry — additional threshold adjustment provides no further benefit.

---

## 2.5 SVM Summary

::: {#14fdab29 .cell execution_count=13}
``` {.python .cell-code}
all_results += [
    eval_model('SVM Linear', y_pred_svm_lin,     y_prob_svm_lin),
    eval_model('SVM Linear', y_pred_svm_lin_opt, y_prob_svm_lin, threshold=opt_t),
    eval_model('SVM RBF',    y_pred_svm_rbf,     y_prob_svm_rbf),
    eval_model('SVM RBF',    y_pred_svm_rbf_opt, y_prob_svm_rbf, threshold=opt_t),
    eval_model('SVM RBF (class_weight)',     y_pred_svm_cw,     y_prob_svm_cw),
    eval_model('SVM RBF (class_weight)',     y_pred_svm_cw_opt, y_prob_svm_cw,
               threshold=opt_t),
]
all_probs['SVM Linear']              = y_prob_svm_lin
all_probs['SVM RBF']                 = y_prob_svm_rbf
all_probs['SVM RBF (class_weight)']  = y_prob_svm_cw

svm_summary = pd.DataFrame([
    eval_model('Linear SVM (t=0.50)',             y_pred_svm_lin,     y_prob_svm_lin),
    eval_model(f'Linear SVM (t={opt_t})',         y_pred_svm_lin_opt, y_prob_svm_lin),
    eval_model('RBF SVM (t=0.50)',                y_pred_svm_rbf,     y_prob_svm_rbf),
    eval_model(f'RBF SVM (t={opt_t})',            y_pred_svm_rbf_opt, y_prob_svm_rbf),
    eval_model('RBF SVM class_weight (t=0.50)',   y_pred_svm_cw,      y_prob_svm_cw),
    eval_model(f'RBF SVM class_weight (t={opt_t})', y_pred_svm_cw_opt, y_prob_svm_cw),
]).set_index('Model').round(4)

display(svm_summary)
```

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Recall (Default)</th>
      <th>Precision (Default)</th>
      <th>F1 (Default)</th>
      <th>ROC-AUC</th>
      <th>Expected Cost</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear SVM (t=0.50)</th>
      <td>0.8042</td>
      <td>0.2446</td>
      <td>0.6537</td>
      <td>0.3560</td>
      <td>0.7394</td>
      <td>0.8642</td>
    </tr>
    <tr>
      <th>Linear SVM (t=0.2)</th>
      <td>0.7031</td>
      <td>0.6504</td>
      <td>0.3959</td>
      <td>0.4922</td>
      <td>0.7394</td>
      <td>0.6062</td>
    </tr>
    <tr>
      <th>RBF SVM (t=0.50)</th>
      <td>0.8094</td>
      <td>0.2853</td>
      <td>0.6605</td>
      <td>0.3985</td>
      <td>0.7082</td>
      <td>0.8230</td>
    </tr>
    <tr>
      <th>RBF SVM (t=0.2)</th>
      <td>0.8030</td>
      <td>0.4380</td>
      <td>0.5714</td>
      <td>0.4959</td>
      <td>0.7082</td>
      <td>0.6943</td>
    </tr>
    <tr>
      <th>RBF SVM class_weight (t=0.50)</th>
      <td>0.6771</td>
      <td>0.7167</td>
      <td>0.3786</td>
      <td>0.4955</td>
      <td>0.7576</td>
      <td>0.5736</td>
    </tr>
    <tr>
      <th>RBF SVM class_weight (t=0.2)</th>
      <td>0.6870</td>
      <td>0.7022</td>
      <td>0.3860</td>
      <td>0.4981</td>
      <td>0.7576</td>
      <td>0.5766</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


### Interpretation

* The summary table confirms that the **cost-sensitive RBF SVM at t = 0.50** delivers the best expected cost (**0.5736**) among all SVM configurations — and improves over Part 1's best result (LDA at t = 0.20, expected cost 0.599).
* The linear SVM and the LDA from Part 1 are nearly interchangeable in ROC-AUC (0.7394 vs 0.7401), reinforcing that the discriminative signal in this dataset is largely linear.
* The unweighted RBF SVM performs worst overall (ROC-AUC = 0.7082), indicating that a naïve kernel mapping with default-threshold evaluation does not help on this tabular dataset. The feature space is already standardised and moderately high-dimensional; the kernel trick adds capacity the data cannot support.
* A key practical insight: for SVMs on this problem, **class weighting is more effective than threshold tuning**. The class-weighted model achieves similar expected cost at both thresholds (0.5736 vs 0.5766), meaning the learner has already absorbed the cost asymmetry during training.

---

# 3. Tree-Based Methods

## 3.1 Decision Tree with Pruning

### Theory

A classification tree partitions the feature space into rectangular regions by recursively applying binary splits that minimise node impurity (Gini index). The full (unpruned) tree grows until every leaf is pure or contains fewer than `min_samples_split` observations — this typically produces a very deep, overfit tree. **Cost-complexity pruning** addresses this by adding a regularisation penalty $\alpha \cdot |T|$ (number of leaves) to the training impurity: as $\alpha$ increases, leaves are progressively collapsed back toward the root. Cross-validation over a grid of $\alpha$ values identifies the level of pruning that maximises generalisation.

### Fit & Prune

::: {#848c8ca9 .cell execution_count=14}
``` {.python .cell-code}
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Grow the largest possible tree
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train_scaled, y_train)

# Cost-complexity pruning path
path      = dt_full.cost_complexity_pruning_path(X_train_scaled, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # remove the last entry (trivial root-only tree)

print(f"Total alpha values in pruning path: {len(ccp_alphas)}")
print(f"Alpha range: [{ccp_alphas.min():.6f}, {ccp_alphas.max():.6f}]")
```

::: {.cell-output .cell-output-stdout}
```
Total alpha values in pruning path: 1312
Alpha range: [0.000000, 0.010897]
```
:::
:::


::: {#e1e4d397 .cell execution_count=15}
``` {.python .cell-code}
# Sample 30 evenly-spaced alpha values to keep CV cost manageable
n_sample   = min(30, len(ccp_alphas))
alpha_idx  = np.unique(np.linspace(0, len(ccp_alphas) - 1, n_sample).astype(int))
alpha_grid = ccp_alphas[alpha_idx]

cv_dt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_mean, cv_auc_sd = [], []

for alpha in alpha_grid:
    scores = cross_val_score(
        DecisionTreeClassifier(ccp_alpha=alpha, random_state=42),
        X_train_scaled, y_train,
        cv=cv_dt, scoring='roc_auc', n_jobs=-1
    )
    cv_auc_mean.append(scores.mean())
    cv_auc_sd.append(scores.std())

cv_auc_mean = np.array(cv_auc_mean)
cv_auc_sd   = np.array(cv_auc_sd)
best_alpha  = alpha_grid[np.argmax(cv_auc_mean)]

print(f"Best alpha: {best_alpha:.6f}  (CV AUC = {cv_auc_mean.max():.4f})")
```

::: {.cell-output .cell-output-stdout}
```
Best alpha: 0.000177  (CV AUC = 0.7313)
```
:::
:::


::: {#5896ec27 .cell execution_count=16}
``` {.python .cell-code}
# Pruning curve: CV AUC vs alpha
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(alpha_grid, cv_auc_mean, marker='o', color='steelblue', lw=1.5, label='Mean CV AUC')
ax.fill_between(alpha_grid,
                cv_auc_mean - cv_auc_sd,
                cv_auc_mean + cv_auc_sd,
                alpha=0.2, color='steelblue')
ax.axvline(best_alpha, color='red', linestyle='--', label=f'Best α = {best_alpha:.5f}')
ax.set_title('Decision Tree: CV ROC-AUC vs Pruning Strength (α)')
ax.set_xlabel('ccp_alpha (α)')
ax.set_ylabel('Mean 5-fold CV ROC-AUC')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-17-output-1.png){width=853 height=372}
:::
:::


::: {#d8b2c4d8 .cell execution_count=17}
``` {.python .cell-code}
# Fit pruned tree on full training set
dt_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
dt_pruned.fit(X_train_scaled, y_train)

print(f"Pruned tree depth   : {dt_pruned.get_depth()}")
print(f"Pruned tree leaves  : {dt_pruned.get_n_leaves()}")

y_pred_dt     = dt_pruned.predict(X_test_scaled)
y_prob_dt     = dt_pruned.predict_proba(X_test_scaled)[:, 1]
y_pred_dt_opt = (y_prob_dt >= opt_t).astype(int)
auc_dt        = roc_auc_score(y_test, y_prob_dt)

print("\n=== Pruned Decision Tree: Classification Report ===")
print(classification_report(y_test, y_pred_dt,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_dt:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_dt):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_dt_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
Pruned tree depth   : 14
Pruned tree leaves  : 83

=== Pruned Decision Tree: Classification Report ===
              precision    recall  f1-score   support

  No Default       0.84      0.95      0.89      7009
     Default       0.64      0.34      0.45      1991

    accuracy                           0.81      9000
   macro avg       0.74      0.64      0.67      9000
weighted avg       0.79      0.81      0.79      9000

ROC-AUC: 0.7568
Expected cost (t=0.50): 0.7698
Expected cost (t=0.2): 0.5981
```
:::
:::


::: {#bc1cfff5 .cell execution_count=18}
``` {.python .cell-code}
# Tree diagram — limit depth for legibility
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('Decision Tree (pruned) — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_dt, ax=axes[1],
                                 name=f'Decision Tree (AUC = {auc_dt:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('Decision Tree (pruned) — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-19-output-1.png){width=1014 height=372}
:::
:::


::: {#978db463 .cell execution_count=19}
``` {.python .cell-code}
# Visualise top 3 levels of the pruned tree
plt.figure(figsize=(18, 6))
plot_tree(dt_pruned, max_depth=3,
          feature_names=X_train_scaled.columns.tolist(),
          class_names=['No Default', 'Default'],
          filled=True, rounded=True, fontsize=8)
plt.title('Pruned Decision Tree (top 3 levels shown)')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-20-output-1.png){width=1715 height=566}
:::
:::


### Interpretation

* The pruning path contains **1 312 candidate α values**; cross-validation selects **α = 0.000177**, producing a tree with **14 levels** and **83 leaves**. This is substantially pruned from the full tree but still large enough to capture complex interactions. The apparent gap on the right side of the pruning curve is an artifact of the sampling strategy: the 30 evaluation points are drawn at evenly-spaced *indices* along the path, but the α values themselves are not uniformly distributed — most are densely clustered near zero (each corresponding to removing a single leaf) and a few at the far end represent collapsing large subtrees in one step. The region of interest around the optimum is therefore sampled densely, while the high-α tail is sampled coarsely.
* The pruned tree achieves a **ROC-AUC of 0.7568**, already competitive with the SVMs. At the standard threshold, recall on defaulters is **34.3 %** (F1 = 0.45), and at the cost-optimal threshold 0.20, recall rises to **65.5 %** with expected cost dropping to **0.5981**.
* The tree diagram reveals that the first split is on a repayment-status feature, consistent with the variable importance patterns observed in Part 1's EDA. The top 3 levels use status and bill amount variables almost exclusively.
* A single tree is inherently interpretable — one can trace any prediction through the decision path — but its **high variance** (sensitivity to the specific training sample) is the main limitation. The next sections address this with ensemble methods.

---

## 3.2 Bagging

### Theory

Bootstrap Aggregating (Bagging) reduces variance by fitting $B$ independent trees on different bootstrap resamples of the training data and averaging their predictions. Because each tree uses all $p$ features at every split (`max_features=1.0`), the trees are highly correlated. Bagging still reduces variance substantially compared to a single tree, but the correlation limits the gain. The **out-of-bag (OOB) score** is a built-in, cost-free estimate of generalisation error: each tree is evaluated on the ≈37 % of training observations it did not see during fitting.

### Fit & Evaluate

::: {#14d7fd5f .cell execution_count=20}
``` {.python .cell-code}
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=150,
    max_features=1.0,   # all features → pure bagging (not random forest)
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
bag.fit(X_train_scaled, y_train)

y_pred_bag     = bag.predict(X_test_scaled)
y_prob_bag     = bag.predict_proba(X_test_scaled)[:, 1]
y_pred_bag_opt = (y_prob_bag >= opt_t).astype(int)
auc_bag        = roc_auc_score(y_test, y_prob_bag)

print(f"OOB accuracy : {bag.oob_score_:.4f}")
print("\n=== Bagging (150 trees): Classification Report ===")
print(classification_report(y_test, y_pred_bag,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_bag:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_bag):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_bag_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
OOB accuracy : 0.8144

=== Bagging (150 trees): Classification Report ===
              precision    recall  f1-score   support

  No Default       0.84      0.94      0.89      7009
     Default       0.64      0.38      0.47      1991

    accuracy                           0.82      9000
   macro avg       0.74      0.66      0.68      9000
weighted avg       0.80      0.82      0.80      9000

ROC-AUC: 0.7544
Expected cost (t=0.50): 0.7367
Expected cost (t=0.2): 0.5926
```
:::
:::


::: {#b745cf30 .cell execution_count=21}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_bag),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('Bagging — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_bag, ax=axes[1],
                                 name=f'Bagging (AUC = {auc_bag:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('Bagging — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-22-output-1.png){width=974 height=372}
:::
:::


### Interpretation

* Bagging with 150 trees achieves an **OOB accuracy of 81.4 %** and a test **ROC-AUC of 0.7544**. Recall on defaulters at the standard threshold is **37.6 %** — a modest improvement over the single pruned tree (34.3 %), reflecting the variance reduction from bootstrap aggregation.
* At the cost-optimal threshold 0.20, recall rises to **70.8 %** with expected cost **0.5926**, which already improves on Part 1's best result (LDA at t = 0.20, expected cost 0.599).
* Recall and expected cost improve over the single tree at both thresholds, but **ROC-AUC drops slightly (0.7544 vs 0.7568)**. This is because all bagging trees use the full feature set at each split, so the dominant features (repayment status variables) appear at the top of nearly every tree. The resulting high inter-tree correlation limits the variance reduction, and the ensemble's ranking quality does not improve on that of the single pruned tree.

---

## 3.3 Random Forest

### Theory

A Random Forest decorrelates the Bagging ensemble by restricting each split to a random subset of $m = \texttt{max\_features}$ candidate predictors. Because trees no longer always pick the same dominant features at the top splits, they become more diverse and their average predictions are less correlated — reducing variance further than Bagging. The canonical default for classification is $m = \lfloor\sqrt{p}\rfloor$. We tune $m$ via OOB error to check whether a different value works better on this dataset.

### Tuning max_features

::: {#b9c98b1b .cell execution_count=22}
``` {.python .cell-code}
from sklearn.ensemble import RandomForestClassifier

p        = X_train_scaled.shape[1]
sqrt_p   = int(np.sqrt(p))
mtry_candidates = sorted(set([
    max(1, sqrt_p - 2), max(1, sqrt_p - 1),
    sqrt_p, sqrt_p + 1, sqrt_p + 2,
    p // 3, p // 2
]))

oob_rows = []
for m in mtry_candidates:
    rf_tmp = RandomForestClassifier(
        n_estimators=100, max_features=m,
        oob_score=True, random_state=42, n_jobs=-1
    )
    rf_tmp.fit(X_train_scaled, y_train)
    oob_rows.append({'max_features': m, 'OOB_error': 1 - rf_tmp.oob_score_})

oob_df     = pd.DataFrame(oob_rows)
best_m     = oob_df.loc[oob_df['OOB_error'].idxmin(), 'max_features']
best_m     = int(best_m)

print("OOB error vs max_features:")
display(oob_df)
print(f"\nBest max_features: {best_m}  (sqrt(p) = {sqrt_p}, p = {p})")
```

::: {.cell-output .cell-output-stdout}
```
OOB error vs max_features:
```
:::

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_features</th>
      <th>OOB_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.186143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.183857</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>0.184143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.185714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.184286</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>0.184381</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12</td>
      <td>0.185762</td>
    </tr>
  </tbody>
</table>
</div>
```
:::

::: {.cell-output .cell-output-stdout}
```

Best max_features: 3  (sqrt(p) = 4, p = 24)
```
:::
:::


::: {#8e93b73b .cell execution_count=23}
``` {.python .cell-code}
plt.figure(figsize=(7, 4))
plt.plot(oob_df['max_features'], oob_df['OOB_error'], marker='o', color='steelblue')
plt.axvline(best_m, color='red', linestyle='--', label=f'Best m = {best_m}')
plt.title('Random Forest: OOB Error vs max_features')
plt.xlabel('max_features (m)')
plt.ylabel('OOB Error (1 − OOB accuracy)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-24-output-1.png){width=661 height=372}
:::
:::


### Fit & Evaluate

::: {#9a379f7e .cell execution_count=24}
``` {.python .cell-code}
rf_best = RandomForestClassifier(
    n_estimators=200, max_features=best_m,
    oob_score=True, random_state=42, n_jobs=-1
)
rf_best.fit(X_train_scaled, y_train)

y_pred_rf     = rf_best.predict(X_test_scaled)
y_prob_rf     = rf_best.predict_proba(X_test_scaled)[:, 1]
y_pred_rf_opt = (y_prob_rf >= opt_t).astype(int)
auc_rf        = roc_auc_score(y_test, y_prob_rf)

print(f"OOB accuracy: {rf_best.oob_score_:.4f}")
print("\n=== Random Forest: Classification Report ===")
print(classification_report(y_test, y_pred_rf,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_rf:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_rf):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_rf_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
OOB accuracy: 0.8177

=== Random Forest: Classification Report ===
              precision    recall  f1-score   support

  No Default       0.84      0.94      0.89      7009
     Default       0.64      0.37      0.47      1991

    accuracy                           0.81      9000
   macro avg       0.74      0.65      0.68      9000
weighted avg       0.80      0.81      0.79      9000

ROC-AUC: 0.7599
Expected cost (t=0.50): 0.7458
Expected cost (t=0.2): 0.5884
```
:::
:::


::: {#50596710 .cell execution_count=25}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('Random Forest — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_rf, ax=axes[1],
                                 name=f'Random Forest (AUC = {auc_rf:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('Random Forest — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-26-output-1.png){width=975 height=372}
:::
:::


### Variable Importance

::: {#5a20c96d .cell execution_count=26}
``` {.python .cell-code}
importance_df = pd.DataFrame({
    'Feature':    X_train_scaled.columns,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 7))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.title('Random Forest — Variable Importance (Mean Decrease in Gini)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-27-output-1.png){width=854 height=660}
:::
:::


### Interpretation

* OOB tuning selects **max_features = 3** (below the $\lfloor\sqrt{p}\rfloor = 4$ default), meaning each split considers only 3 of the 24 features. This aggressive restriction maximises tree diversity and pushes the OOB error slightly below the alternatives.
* The OOB error differences across max_features candidates are small (ranging from 0.1839 to 0.1861), indicating that the forest is fairly robust to this hyperparameter on this dataset. Nevertheless, the small gain from m = 3 over m = 4 is consistent: restricting the number of candidates forces the trees to explore less dominant features, decorrelating the ensemble.
* The final Random Forest (200 trees, m = 3) achieves **ROC-AUC = 0.7599** and **OOB accuracy = 81.8 %**. At the cost-optimal threshold, recall rises to **69.1 %** with expected cost **0.5884** — a modest improvement over Bagging (0.5926).
* The **variable importance plot** reveals a clear hierarchy: repayment-status features (`status_sep` through `status_apr`) and bill/payment amounts dominate, while demographic features (`gender`, `marital_status`, `education`) contribute very little. This aligns with Part 1's EDA findings (Section 3) where payment history was the strongest predictor of default.
* Compared to Bagging, the Random Forest's AUC improvement is modest (+0.0055), confirming that the dominant signal in this dataset is concentrated in a few strong predictors. Feature randomisation helps, but the gains are limited when most predictive power comes from a small subset of features.

---

## 3.4 Gradient Boosting (GBM)

### Theory

Gradient Boosting builds an additive ensemble sequentially: each new tree fits the **pseudo-residuals** (negative gradient of the loss) of the current ensemble rather than the original target. The ensemble grows as:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \, h_m(\mathbf{x})$$

where $\eta$ (learning rate, `learning_rate`) controls the contribution of each tree and $h_m$ is a shallow tree. Key hyperparameters are: `n_estimators` (number of boosting rounds), `learning_rate`, `max_depth` (tree complexity), and `subsample` (stochastic boosting). Unlike Bagging and RF, adding more trees **can** overfit, so the number of rounds must be tuned.

### Fit & Evaluate

::: {#800b4854 .cell execution_count=27}
``` {.python .cell-code}
from sklearn.ensemble import GradientBoostingClassifier

gbm_param_grid = {
    'n_estimators':    [100, 200],
    'max_depth':       [2, 3],
    'learning_rate':   [0.05, 0.10],
    'subsample':       [0.8],
}

cv_gbm = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gbm_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gbm_param_grid,
    cv=cv_gbm,
    scoring='roc_auc',
    n_jobs=-1,
)
gbm_gs.fit(X_train_scaled, y_train)

print(f"Best GBM params : {gbm_gs.best_params_}")
print(f"Best CV AUC     : {gbm_gs.best_score_:.4f}")

gbm_best = gbm_gs.best_estimator_

y_pred_gbm     = gbm_best.predict(X_test_scaled)
y_prob_gbm     = gbm_best.predict_proba(X_test_scaled)[:, 1]
y_pred_gbm_opt = (y_prob_gbm >= opt_t).astype(int)
auc_gbm        = roc_auc_score(y_test, y_prob_gbm)

print("\n=== GBM (best params): Classification Report ===")
print(classification_report(y_test, y_pred_gbm,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_gbm:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_gbm):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_gbm_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
Best GBM params : {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}
Best CV AUC     : 0.7815

=== GBM (best params): Classification Report ===
              precision    recall  f1-score   support

  No Default       0.84      0.95      0.89      7009
     Default       0.66      0.36      0.46      1991

    accuracy                           0.82      9000
   macro avg       0.75      0.65      0.68      9000
weighted avg       0.80      0.82      0.80      9000

ROC-AUC: 0.7809
Expected cost (t=0.50): 0.7524
Expected cost (t=0.2): 0.5604
```
:::
:::


::: {#43b3f3c3 .cell execution_count=28}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_gbm),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('GBM — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_gbm, ax=axes[1],
                                 name=f'GBM (AUC = {auc_gbm:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('GBM — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-29-output-1.png){width=974 height=372}
:::
:::


### Interpretation

* GridSearchCV selects **learning_rate = 0.05, max_depth = 3, n_estimators = 200, subsample = 0.8** with a CV AUC of **0.7815** — the highest AUC achieved by any model so far.
* On the test set, GBM achieves **ROC-AUC = 0.7809**, an improvement of **+0.0210** over Random Forest (0.7599) and **+0.0408** over LDA (0.7401). This confirms that the sequential, residual-fitting nature of boosting extracts additional discriminative signal that bagging-based methods miss.
* At the cost-optimal threshold 0.20, recall reaches **66.1 %** with the **lowest expected cost of any tree-based method (0.5604)** — a reduction of 0.039 expected cost units compared to Part 1's LDA benchmark (0.599).
* The selected configuration uses shallow trees (max_depth = 3) and a low learning rate (0.05), consistent with the standard boosting recipe: many weak learners combined cautiously. Stochastic boosting (subsample = 0.8) adds variance reduction similar to bagging.

---

## 3.5 XGBoost

### Theory

XGBoost (Extreme Gradient Boosting) is an optimised implementation of gradient boosting that adds: (i) a **regularisation term** ($L_1$ via `reg_alpha` and $L_2$ via `reg_lambda`) directly in the objective to control tree complexity; (ii) **column subsampling** (`colsample_bytree`) analogous to Random Forest's feature randomisation, further reducing correlation between trees; (iii) efficient approximate split-finding with built-in support for `eval_set` and early stopping to automatically select the number of rounds. These additions typically improve both performance and overfitting resistance compared to standard GBM.

### Fit & Evaluate

::: {#14beccc6 .cell execution_count=29}
``` {.python .cell-code}
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=400,
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(
    X_train_scaled, y_train,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False,
)

best_round = xgb.best_iteration
print(f"Best number of boosting rounds (early stopping): {best_round}")

y_pred_xgb     = xgb.predict(X_test_scaled)
y_prob_xgb     = xgb.predict_proba(X_test_scaled)[:, 1]
y_pred_xgb_opt = (y_prob_xgb >= opt_t).astype(int)
auc_xgb        = roc_auc_score(y_test, y_prob_xgb)

print("\n=== XGBoost: Classification Report ===")
print(classification_report(y_test, y_pred_xgb,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_xgb:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_test, y_pred_xgb):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_test, y_pred_xgb_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
Best number of boosting rounds (early stopping): 183

=== XGBoost: Classification Report ===
              precision    recall  f1-score   support

  No Default       0.84      0.95      0.89      7009
     Default       0.66      0.36      0.46      1991

    accuracy                           0.82      9000
   macro avg       0.75      0.65      0.68      9000
weighted avg       0.80      0.82      0.80      9000

ROC-AUC: 0.7809
Expected cost (t=0.50): 0.7530
Expected cost (t=0.2): 0.5622
```
:::
:::


::: {#43532e7e .cell execution_count=30}
``` {.python .cell-code}
# XGBoost learning curve (train vs test AUC per boosting round)
xgb_evals = xgb.evals_result()
train_auc  = xgb_evals['validation_0']['auc']
test_auc   = xgb_evals['validation_1']['auc']
rounds     = range(1, len(train_auc) + 1)

plt.figure(figsize=(9, 4))
plt.plot(rounds, train_auc, label='Train AUC',      color='steelblue', lw=1.5)
plt.plot(rounds, test_auc,  label='Test AUC',       color='tab:orange', lw=1.5)
plt.axvline(best_round, color='red', linestyle='--',
            label=f'Early-stop round = {best_round}')
plt.title('XGBoost: ROC-AUC per Boosting Round')
plt.xlabel('Boosting round')
plt.ylabel('ROC-AUC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-31-output-1.png){width=853 height=372}
:::
:::


::: {#be394437 .cell execution_count=31}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_xgb),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('XGBoost — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_test, y_prob_xgb, ax=axes[1],
                                 name=f'XGBoost (AUC = {auc_xgb:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('XGBoost — ROC Curve')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-32-output-1.png){width=974 height=372}
:::
:::


### Interpretation

* XGBoost selects **183 boosting rounds** via early stopping (from a maximum of 400), confirming that the built-in regularisation and monitoring prevent overfitting effectively.
* Test performance is virtually identical to GBM: **ROC-AUC = 0.7809**, accuracy = 81.8 %, recall at t = 0.50 = 35.5 %. At the cost-optimal threshold, expected cost is **0.5622**, within 0.002 of GBM's 0.5604.
* The learning curve shows training AUC climbing steadily while test AUC plateaus after approximately 100 rounds, with the gap widening slightly — classic mild overfitting behaviour controlled by early stopping.
* The near-parity with GBM is expected: both use the same depth (3), a similar learning rate (0.05), and stochastic row sampling (0.8). XGBoost's column subsampling (0.8) and $L_2$ regularisation provide a minor additional hedge against overfitting but do not materially change the outcome on this moderately-sized dataset.

---

## 3.6 Tree-Based Methods Summary

::: {#339cf2ee .cell execution_count=32}
``` {.python .cell-code}
all_results += [
    eval_model('Decision Tree', y_pred_dt,     y_prob_dt),
    eval_model('Decision Tree', y_pred_dt_opt, y_prob_dt,  threshold=opt_t),
    eval_model('Bagging',       y_pred_bag,    y_prob_bag),
    eval_model('Bagging',       y_pred_bag_opt,y_prob_bag, threshold=opt_t),
    eval_model('Random Forest', y_pred_rf,     y_prob_rf),
    eval_model('Random Forest', y_pred_rf_opt, y_prob_rf,  threshold=opt_t),
    eval_model('GBM',           y_pred_gbm,    y_prob_gbm),
    eval_model('GBM',           y_pred_gbm_opt,y_prob_gbm, threshold=opt_t),
    eval_model('XGBoost',       y_pred_xgb,    y_prob_xgb),
    eval_model('XGBoost',       y_pred_xgb_opt,y_prob_xgb, threshold=opt_t),
]
all_probs['Decision Tree'] = y_prob_dt
all_probs['Bagging']       = y_prob_bag
all_probs['Random Forest'] = y_prob_rf
all_probs['GBM']           = y_prob_gbm
all_probs['XGBoost']       = y_prob_xgb

tree_summary = pd.DataFrame([
    eval_model('Decision Tree (t=0.50)',     y_pred_dt,     y_prob_dt),
    eval_model(f'Decision Tree (t={opt_t})', y_pred_dt_opt, y_prob_dt),
    eval_model('Bagging (t=0.50)',           y_pred_bag,    y_prob_bag),
    eval_model(f'Bagging (t={opt_t})',       y_pred_bag_opt,y_prob_bag),
    eval_model('Random Forest (t=0.50)',     y_pred_rf,     y_prob_rf),
    eval_model(f'Random Forest (t={opt_t})', y_pred_rf_opt, y_prob_rf),
    eval_model('GBM (t=0.50)',               y_pred_gbm,    y_prob_gbm),
    eval_model(f'GBM (t={opt_t})',           y_pred_gbm_opt,y_prob_gbm),
    eval_model('XGBoost (t=0.50)',           y_pred_xgb,    y_prob_xgb),
    eval_model(f'XGBoost (t={opt_t})',       y_pred_xgb_opt,y_prob_xgb),
]).set_index('Model').round(4)

display(tree_summary)
```

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Recall (Default)</th>
      <th>Precision (Default)</th>
      <th>F1 (Default)</th>
      <th>ROC-AUC</th>
      <th>Expected Cost</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Decision Tree (t=0.50)</th>
      <td>0.8120</td>
      <td>0.3425</td>
      <td>0.6404</td>
      <td>0.4463</td>
      <td>0.7568</td>
      <td>0.7698</td>
    </tr>
    <tr>
      <th>Decision Tree (t=0.2)</th>
      <td>0.7072</td>
      <td>0.6549</td>
      <td>0.4010</td>
      <td>0.4974</td>
      <td>0.7568</td>
      <td>0.5981</td>
    </tr>
    <tr>
      <th>Bagging (t=0.50)</th>
      <td>0.8158</td>
      <td>0.3757</td>
      <td>0.6432</td>
      <td>0.4743</td>
      <td>0.7544</td>
      <td>0.7367</td>
    </tr>
    <tr>
      <th>Bagging (t=0.2)</th>
      <td>0.6661</td>
      <td>0.7077</td>
      <td>0.3677</td>
      <td>0.4839</td>
      <td>0.7544</td>
      <td>0.5926</td>
    </tr>
    <tr>
      <th>Random Forest (t=0.50)</th>
      <td>0.8147</td>
      <td>0.3666</td>
      <td>0.6420</td>
      <td>0.4668</td>
      <td>0.7599</td>
      <td>0.7458</td>
    </tr>
    <tr>
      <th>Random Forest (t=0.2)</th>
      <td>0.6853</td>
      <td>0.6906</td>
      <td>0.3829</td>
      <td>0.4927</td>
      <td>0.7599</td>
      <td>0.5884</td>
    </tr>
    <tr>
      <th>GBM (t=0.50)</th>
      <td>0.8173</td>
      <td>0.3561</td>
      <td>0.6620</td>
      <td>0.4631</td>
      <td>0.7809</td>
      <td>0.7524</td>
    </tr>
    <tr>
      <th>GBM (t=0.2)</th>
      <td>0.7396</td>
      <td>0.6610</td>
      <td>0.4409</td>
      <td>0.5289</td>
      <td>0.7809</td>
      <td>0.5604</td>
    </tr>
    <tr>
      <th>XGBoost (t=0.50)</th>
      <td>0.8177</td>
      <td>0.3551</td>
      <td>0.6645</td>
      <td>0.4628</td>
      <td>0.7809</td>
      <td>0.7530</td>
    </tr>
    <tr>
      <th>XGBoost (t=0.2)</th>
      <td>0.7369</td>
      <td>0.6620</td>
      <td>0.4374</td>
      <td>0.5268</td>
      <td>0.7809</td>
      <td>0.5622</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


::: {#37e7329d .cell execution_count=33}
``` {.python .cell-code}
# Combined ROC curve for all tree-based methods
fig, ax = plt.subplots(figsize=(8, 6))

for name, prob in [('Decision Tree', y_prob_dt), ('Bagging',  y_prob_bag),
                   ('Random Forest', y_prob_rf),  ('GBM',      y_prob_gbm),
                   ('XGBoost',       y_prob_xgb)]:
    RocCurveDisplay.from_predictions(
        y_test, prob, ax=ax,
        name=f'{name} (AUC = {roc_auc_score(y_test, prob):.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.set_title('ROC Curves: Tree-Based Methods')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-34-output-1.png){width=559 height=564}
:::
:::


### Interpretation

* The summary table shows a clear hierarchy among tree-based methods. **GBM and XGBoost tie for the best ROC-AUC (0.7809)**, followed by Random Forest (0.7599), Decision Tree (0.7568), and Bagging (0.7544). The boosting methods' sequential residual-fitting strategy provides a meaningful edge over the parallel averaging of bagging-based ensembles.
* At the cost-optimal threshold 0.20, **GBM achieves the lowest expected cost (0.5604)**, followed closely by XGBoost (0.5622). Both outperform all SVM variants and all Part 1 models.
* The ROC overlay confirms the ranking visually: the GBM and XGBoost curves dominate the upper-left region, while the single decision tree curve lies noticeably below the ensembles.
* An interesting pattern: at the standard threshold 0.50, all tree methods have very similar accuracy (~81–82 %) and recall (~34–38 %). The differences become apparent only at the cost-optimal threshold or in ROC-AUC, underscoring the importance of evaluating beyond accuracy on imbalanced datasets.
* Bagging and Random Forest occupy a middle tier: they reduce variance relative to the single tree but cannot correct the bias toward the majority class without threshold adjustment. Boosting, by contrast, explicitly targets the residual error, making it more effective at identifying the hard-to-classify defaulters.

---

# 4. Neural Networks (MLP)

## 4.1 Theory

A **Multi-Layer Perceptron (MLP)** is a feedforward neural network composed of an input layer, one or more hidden layers, and an output layer. Each layer applies an affine transformation followed by a non-linear activation:

$$\mathbf{h}^{(l)} = \text{ReLU}\!\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

For binary classification with a single output neuron, the raw output (logit) $z = \mathbf{w}^T\mathbf{h} + b$ is converted to a probability via $\hat{p} = \sigma(z) = (1 + e^{-z})^{-1}$. Training minimises **binary cross-entropy** using stochastic gradient descent (SGD) with **backpropagation** to compute exact gradients.

Key hyperparameters: hidden layer widths, learning rate, dropout probability (regularisation), batch size, and number of training epochs. We use the **Adam** optimiser, which adapts the learning rate per parameter and typically converges faster than vanilla SGD.

---

## 4.2 Architecture & Data Preparation

::: {#17171494 .cell execution_count=34}
``` {.python .cell-code}
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── TensorDatasets ────────────────────────────────────────────────────────────
def make_tensor_loaders(X_tr, y_tr, X_te, y_te, batch_size=256,
                        val_frac=0.15, seed=42):
    """
    Splits training data into train/val, creates TensorDatasets and DataLoaders.
    val_frac: fraction of training rows reserved for validation.
    """
    rng_t = torch.Generator().manual_seed(seed)
    n_val = int(len(X_tr) * val_frac)
    n_tr  = len(X_tr) - n_val
    perm  = torch.randperm(len(X_tr), generator=rng_t)

    idx_tr  = perm[:n_tr]
    idx_val = perm[n_tr:]

    X_tr_t  = torch.tensor(X_tr.values,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr.values,  dtype=torch.float32)
    X_te_t  = torch.tensor(X_te.values,  dtype=torch.float32)
    y_te_t  = torch.tensor(y_te.values,  dtype=torch.float32)

    ds_train = TensorDataset(X_tr_t[idx_tr],  y_tr_t[idx_tr])
    ds_val   = TensorDataset(X_tr_t[idx_val], y_tr_t[idx_val])
    ds_test  = TensorDataset(X_te_t,           y_te_t)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False)
    loader_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False)

    return loader_train, loader_val, loader_test

loader_train, loader_val, loader_test = make_tensor_loaders(
    X_train_scaled, y_train, X_test_scaled, y_test
)
print(f"Train batches: {len(loader_train)} | "
      f"Val batches: {len(loader_val)} | "
      f"Test batches: {len(loader_test)}")
```

::: {.cell-output .cell-output-stdout}
```
Device: cpu
Train batches: 70 | Val batches: 13 | Test batches: 36
```
:::
:::


::: {#e64d2ee1 .cell execution_count=35}
``` {.python .cell-code}
# ── Model definition ──────────────────────────────────────────────────────────
class CreditDefaultMLP(nn.Module):
    """
    Feedforward MLP for binary credit-default classification.
    Output: single logit (no sigmoid — compatible with BCEWithLogitsLoss).
    """
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # (B, 1) → (B,)

input_dim = X_train_scaled.shape[1]
print(f"Input dimension: {input_dim}")

# Quick architecture preview
_model_preview = CreditDefaultMLP(input_dim)
print(_model_preview)
```

::: {.cell-output .cell-output-stdout}
```
Input dimension: 24
CreditDefaultMLP(
  (net): Sequential(
    (0): Linear(in_features=24, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
    (6): Linear(in_features=64, out_features=1, bias=True)
  )
)
```
:::
:::


::: {#e802fed5 .cell execution_count=36}
``` {.python .cell-code}
# ── Training utilities ────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score as _auc

def train_one_epoch(model, loader, criterion, optimiser, device):
    """One forward/backward pass over all training mini-batches."""
    model.train()
    total_loss, total = 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        logits = model(xb)          # raw logits (no sigmoid)
        loss   = criterion(logits, yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(xb)
        total      += len(xb)

    return total_loss / total


def evaluate_loader(model, loader, criterion, device):
    """Return (avg_loss, ROC-AUC) on a DataLoader without gradient tracking."""
    model.eval()
    total_loss, total = 0.0, 0
    all_logits, all_y = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            total      += len(xb)
            all_logits.append(torch.sigmoid(logits).cpu())
            all_y.append(yb.cpu())

    probs  = torch.cat(all_logits).numpy()
    labels = torch.cat(all_y).numpy()
    auc    = _auc(labels, probs)
    return total_loss / total, auc


def fit_mlp(model, loader_train, loader_val, criterion, optimiser,
            epochs, device, verbose_every=5):
    """Full training loop; returns history dict with per-epoch metrics."""
    history = {'train_loss': [], 'val_loss': [],
               'train_auc': [],  'val_auc': []}

    for epoch in range(1, epochs + 1):
        tr_loss                  = train_one_epoch(model, loader_train, criterion,
                                                   optimiser, device)
        tr_loss_eval, tr_auc     = evaluate_loader(model, loader_train, criterion, device)
        val_loss,     val_auc    = evaluate_loader(model, loader_val,   criterion, device)

        history['train_loss'].append(tr_loss_eval)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(tr_auc)
        history['val_auc'].append(val_auc)

        if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"train_loss={tr_loss_eval:.4f}, train_auc={tr_auc:.4f} | "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}")

    return history


def predict_mlp(model, loader, device):
    """Return (y_true, y_prob) numpy arrays from a DataLoader."""
    model.eval()
    all_probs, all_y = [], []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs  = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_y.append(yb)

    return (torch.cat(all_y).numpy().astype(int),
            torch.cat(all_probs).numpy())


def plot_learning_curves(history, title_prefix='MLP'):
    """Plot loss and AUC curves from history dict."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train',      color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Validation', color='tab:orange')
    axes[0].set_title(f'{title_prefix} — Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_auc'], label='Train',      color='steelblue')
    axes[1].plot(epochs, history['val_auc'],   label='Validation', color='tab:orange')
    axes[1].set_title(f'{title_prefix} — ROC-AUC Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```
:::


---

## 4.3 Baseline Model

A first model is fitted with default hyperparameters to establish a baseline before tuning.

::: {#3c27da2c .cell execution_count=37}
``` {.python .cell-code}
torch.manual_seed(42)

baseline_cfg = dict(hidden1=128, hidden2=64, dropout=0.2, lr=1e-3, epochs=20)

baseline_model = CreditDefaultMLP(
    input_dim, hidden1=baseline_cfg['hidden1'],
    hidden2=baseline_cfg['hidden2'], dropout=baseline_cfg['dropout']
).to(device)

# Standard BCE loss (no pos_weight) for baseline
criterion_base = nn.BCEWithLogitsLoss()
optimiser_base = torch.optim.Adam(baseline_model.parameters(),
                                  lr=baseline_cfg['lr'])

baseline_history = fit_mlp(
    baseline_model, loader_train, loader_val,
    criterion_base, optimiser_base,
    epochs=baseline_cfg['epochs'], device=device, verbose_every=5
)
```

::: {.cell-output .cell-output-stdout}
```
Epoch 001/20 | train_loss=0.4463, train_auc=0.7588 | val_loss=0.4574, val_auc=0.7568
Epoch 005/20 | train_loss=0.4280, train_auc=0.7799 | val_loss=0.4434, val_auc=0.7747
Epoch 010/20 | train_loss=0.4225, train_auc=0.7890 | val_loss=0.4402, val_auc=0.7795
Epoch 015/20 | train_loss=0.4192, train_auc=0.7941 | val_loss=0.4421, val_auc=0.7814
Epoch 020/20 | train_loss=0.4136, train_auc=0.7999 | val_loss=0.4385, val_auc=0.7829
```
:::
:::


::: {#e411541f .cell execution_count=38}
``` {.python .cell-code}
plot_learning_curves(baseline_history, title_prefix='Baseline MLP')
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-39-output-1.png){width=1237 height=372}
:::
:::


::: {#218e3e78 .cell execution_count=39}
``` {.python .cell-code}
y_true_base, y_prob_base = predict_mlp(baseline_model, loader_test, device)
y_pred_base     = (y_prob_base >= 0.50).astype(int)
y_pred_base_opt = (y_prob_base >= opt_t).astype(int)
auc_base        = roc_auc_score(y_true_base, y_prob_base)

print("=== Baseline MLP: Classification Report (t=0.50) ===")
print(classification_report(y_true_base, y_pred_base,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_base:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_true_base, y_pred_base):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_true_base, y_pred_base_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
=== Baseline MLP: Classification Report (t=0.50) ===
              precision    recall  f1-score   support

  No Default       0.84      0.94      0.89      7009
     Default       0.64      0.39      0.48      1991

    accuracy                           0.82      9000
   macro avg       0.74      0.66      0.69      9000
weighted avg       0.80      0.82      0.80      9000

ROC-AUC: 0.7716
Expected cost (t=0.50): 0.7248
Expected cost (t=0.2): 0.5659
```
:::
:::


### Interpretation

* The baseline MLP (128 → 64, dropout 0.2, lr = 1e-3, 20 epochs) achieves **ROC-AUC = 0.7716** — already competitive with Random Forest (0.7599) and approaching GBM (0.7809).
* The learning curves show smooth convergence without large train–validation gaps, suggesting the network is not severely overfitting. Validation AUC stabilises near **0.783** by epoch 20.
* At the standard threshold, recall on defaulters is **38.9 %** (F1 = 0.48), marginally better than most tree-based methods at the same threshold. At the cost-optimal threshold 0.20, expected cost is **0.5659**, placing the baseline MLP between GBM and Random Forest.
* These results confirm that even a small, untuned MLP can extract non-linear signal from the standardised tabular features. The next step tests whether larger architectures or different regularisation provide further gains.

---

## 4.4 Hyperparameter Search

We run a compact manual grid over hidden layer widths, dropout, and learning rate. Each configuration is trained for a short number of epochs on a subset of the training data; the winner is then retrained on the full dataset.

::: {#17ea04be .cell execution_count=40}
``` {.python .cell-code}
# Subset of training data for fast search (60% of training samples)
rng_np  = np.random.default_rng(42)
tune_n  = int(len(X_train_scaled) * 0.6)
tune_idx_nn = rng_np.choice(len(X_train_scaled), size=tune_n, replace=False)

X_tune_nn = X_train_scaled.iloc[tune_idx_nn].reset_index(drop=True)
y_tune_nn = y_train.iloc[tune_idx_nn].reset_index(drop=True)

loader_train_tune, loader_val_tune, _ = make_tensor_loaders(
    X_tune_nn, y_tune_nn, X_test_scaled, y_test, batch_size=256
)

hp_grid = [
    {'hidden1': 128, 'hidden2':  64, 'dropout': 0.2, 'lr': 1e-3},
    {'hidden1': 256, 'hidden2': 128, 'dropout': 0.2, 'lr': 1e-3},
    {'hidden1': 128, 'hidden2':  64, 'dropout': 0.3, 'lr': 5e-4},
    {'hidden1': 256, 'hidden2': 128, 'dropout': 0.3, 'lr': 5e-4},
    {'hidden1': 256, 'hidden2':  64, 'dropout': 0.2, 'lr': 1e-3},
    {'hidden1': 128, 'hidden2':  64, 'dropout': 0.1, 'lr': 1e-3},
]

N_QUICK = 10   # epochs per candidate during search

hp_results = []
for i, cfg in enumerate(hp_grid, 1):
    torch.manual_seed(42)
    m = CreditDefaultMLP(input_dim, hidden1=cfg['hidden1'],
                         hidden2=cfg['hidden2'], dropout=cfg['dropout']).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
    opt  = torch.optim.Adam(m.parameters(), lr=cfg['lr'])
    hist = fit_mlp(m, loader_train_tune, loader_val_tune, crit, opt,
                   epochs=N_QUICK, device=device, verbose_every=N_QUICK + 1)
    best_val_auc = max(hist['val_auc'])
    hp_results.append({'config': cfg, 'best_val_auc': best_val_auc})
    print(f"Config {i}/{len(hp_grid)} | val_auc={best_val_auc:.4f} | {cfg}")

hp_results.sort(key=lambda r: r['best_val_auc'], reverse=True)
best_cfg = hp_results[0]['config']

print("\n=== Hyperparameter search ranking ===")
for rank, r in enumerate(hp_results, 1):
    print(f"  {rank}. val_auc={r['best_val_auc']:.4f} | {r['config']}")
print(f"\nBest config: {best_cfg}")
```

::: {.cell-output .cell-output-stdout}
```
Epoch 001/10 | train_loss=1.0829, train_auc=0.7595 | val_loss=1.0817, val_auc=0.7681
Epoch 010/10 | train_loss=1.0170, train_auc=0.7933 | val_loss=1.0329, val_auc=0.7903
Config 1/6 | val_auc=0.7903 | {'hidden1': 128, 'hidden2': 64, 'dropout': 0.2, 'lr': 0.001}
Epoch 001/10 | train_loss=1.0644, train_auc=0.7683 | val_loss=1.0651, val_auc=0.7725
Epoch 010/10 | train_loss=0.9997, train_auc=0.8024 | val_loss=1.0328, val_auc=0.7920
Config 2/6 | val_auc=0.7920 | {'hidden1': 256, 'hidden2': 128, 'dropout': 0.2, 'lr': 0.001}
Epoch 001/10 | train_loss=1.1229, train_auc=0.7520 | val_loss=1.1280, val_auc=0.7563
Epoch 010/10 | train_loss=1.0409, train_auc=0.7803 | val_loss=1.0470, val_auc=0.7826
Config 3/6 | val_auc=0.7826 | {'hidden1': 128, 'hidden2': 64, 'dropout': 0.3, 'lr': 0.0005}
Epoch 001/10 | train_loss=1.0825, train_auc=0.7583 | val_loss=1.0839, val_auc=0.7629
Epoch 010/10 | train_loss=1.0249, train_auc=0.7889 | val_loss=1.0369, val_auc=0.7886
Config 4/6 | val_auc=0.7886 | {'hidden1': 256, 'hidden2': 128, 'dropout': 0.3, 'lr': 0.0005}
Epoch 001/10 | train_loss=1.0713, train_auc=0.7649 | val_loss=1.0665, val_auc=0.7731
Epoch 010/10 | train_loss=1.0077, train_auc=0.7995 | val_loss=1.0372, val_auc=0.7891
Config 5/6 | val_auc=0.7905 | {'hidden1': 256, 'hidden2': 64, 'dropout': 0.2, 'lr': 0.001}
Epoch 001/10 | train_loss=1.0809, train_auc=0.7602 | val_loss=1.0788, val_auc=0.7689
Epoch 010/10 | train_loss=1.0103, train_auc=0.7967 | val_loss=1.0312, val_auc=0.7927
Config 6/6 | val_auc=0.7927 | {'hidden1': 128, 'hidden2': 64, 'dropout': 0.1, 'lr': 0.001}

=== Hyperparameter search ranking ===
  1. val_auc=0.7927 | {'hidden1': 128, 'hidden2': 64, 'dropout': 0.1, 'lr': 0.001}
  2. val_auc=0.7920 | {'hidden1': 256, 'hidden2': 128, 'dropout': 0.2, 'lr': 0.001}
  3. val_auc=0.7905 | {'hidden1': 256, 'hidden2': 64, 'dropout': 0.2, 'lr': 0.001}
  4. val_auc=0.7903 | {'hidden1': 128, 'hidden2': 64, 'dropout': 0.2, 'lr': 0.001}
  5. val_auc=0.7886 | {'hidden1': 256, 'hidden2': 128, 'dropout': 0.3, 'lr': 0.0005}
  6. val_auc=0.7826 | {'hidden1': 128, 'hidden2': 64, 'dropout': 0.3, 'lr': 0.0005}

Best config: {'hidden1': 128, 'hidden2': 64, 'dropout': 0.1, 'lr': 0.001}
```
:::
:::


---

## 4.5 Final Model

The best configuration is retrained on the full training set for a larger number of epochs.

::: {#15d7ec7d .cell execution_count=41}
``` {.python .cell-code}
N_FULL = 40   # full training epochs

torch.manual_seed(42)
mlp_final = CreditDefaultMLP(
    input_dim,
    hidden1=best_cfg['hidden1'],
    hidden2=best_cfg['hidden2'],
    dropout=best_cfg['dropout'],
).to(device)

criterion_final = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
optimiser_final = torch.optim.Adam(mlp_final.parameters(), lr=best_cfg['lr'])

final_history = fit_mlp(
    mlp_final, loader_train, loader_val,
    criterion_final, optimiser_final,
    epochs=N_FULL, device=device, verbose_every=10
)
```

::: {.cell-output .cell-output-stdout}
```
Epoch 001/40 | train_loss=1.0635, train_auc=0.7681 | val_loss=1.0827, val_auc=0.7657
Epoch 010/40 | train_loss=1.0117, train_auc=0.7965 | val_loss=1.0545, val_auc=0.7817
Epoch 020/40 | train_loss=0.9846, train_auc=0.8090 | val_loss=1.0578, val_auc=0.7830
Epoch 030/40 | train_loss=0.9497, train_auc=0.8254 | val_loss=1.0602, val_auc=0.7813
Epoch 040/40 | train_loss=0.9271, train_auc=0.8334 | val_loss=1.0810, val_auc=0.7798
```
:::
:::


::: {#c8f148dd .cell execution_count=42}
``` {.python .cell-code}
plot_learning_curves(final_history, title_prefix='Tuned MLP (final)')
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-43-output-1.png){width=1237 height=372}
:::
:::


::: {#eaee0ecf .cell execution_count=43}
``` {.python .cell-code}
y_true_mlp, y_prob_mlp = predict_mlp(mlp_final, loader_test, device)
y_pred_mlp     = (y_prob_mlp >= 0.50).astype(int)
y_pred_mlp_opt = (y_prob_mlp >= opt_t).astype(int)
auc_mlp        = roc_auc_score(y_true_mlp, y_prob_mlp)

print("=== Tuned MLP: Classification Report (t=0.50) ===")
print(classification_report(y_true_mlp, y_pred_mlp,
                             target_names=['No Default', 'Default']))
print(f"ROC-AUC: {auc_mlp:.4f}")
print(f"Expected cost (t=0.50): {expected_cost(y_true_mlp, y_pred_mlp):.4f}")
print(f"Expected cost (t={opt_t}): {expected_cost(y_true_mlp, y_pred_mlp_opt):.4f}")
```

::: {.cell-output .cell-output-stdout}
```
=== Tuned MLP: Classification Report (t=0.50) ===
              precision    recall  f1-score   support

  No Default       0.89      0.68      0.77      7009
     Default       0.38      0.71      0.50      1991

    accuracy                           0.68      9000
   macro avg       0.64      0.69      0.63      9000
weighted avg       0.78      0.68      0.71      9000

ROC-AUC: 0.7678
Expected cost (t=0.50): 0.5747
Expected cost (t=0.2): 0.6474
```
:::
:::


::: {#36af12d0 .cell execution_count=44}
``` {.python .cell-code}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ConfusionMatrixDisplay(confusion_matrix(y_true_mlp, y_pred_mlp),
                       display_labels=['No Default', 'Default']).plot(
    ax=axes[0], colorbar=False)
axes[0].set_title('Tuned MLP — Confusion Matrix (t=0.50)')

RocCurveDisplay.from_predictions(y_true_mlp, y_prob_mlp, ax=axes[1],
                                 name=f'Tuned MLP (AUC = {auc_mlp:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
axes[1].set_title('Tuned MLP — ROC Curve')
plt.tight_layout()
plt.show()

all_results += [
    eval_model('MLP', y_pred_mlp,     y_prob_mlp),
    eval_model('MLP', y_pred_mlp_opt, y_prob_mlp, threshold=opt_t),
]
all_probs['MLP'] = y_prob_mlp
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-45-output-1.png){width=974 height=372}
:::
:::


### Interpretation

* The hyperparameter search ranks six configurations by validation AUC after 10 quick epochs on the weighted loss. The winning configuration is **hidden1 = 128, hidden2 = 64, dropout = 0.1, lr = 1e-3** (val AUC = 0.7927). Compared to the unweighted run, a smaller and less regularised network now wins: with `pos_weight = 5` amplifying the signal from every positive example, the model needs less dropout to avoid overfitting on the minority class. The top four configs are within 0.0024 of each other, so the ranking should not be over-interpreted. Note that the BCE loss values (~1.0) are not comparable to the baseline (~0.4); `pos_weight = 5` scales the positive-class loss contribution by five, which raises the absolute loss without indicating worse fit.
* Higher learning rates (1e-3 vs 5e-4) consistently outperform lower ones, as before, because 10 epochs are too few for the slower rate to converge. Dropout 0.3 with a slow learning rate remains the weakest configuration.
* The final tuned MLP (128 → 64, 40 epochs, `pos_weight = 5`) achieves **ROC-AUC = 0.7678**, much closer to the baseline's 0.7716 than the previous unweighted run (0.7649). Overfitting is present but mild: training AUC climbs to **0.8334** while validation AUC peaks around epoch 20 at **0.7830** and declines gently to 0.7798 by epoch 40. Validation BCE loss rises from its minimum at epoch 10, confirming the same mild overfitting from a second angle. The train–validation gap is smaller than in the unweighted run, suggesting the cost-weighting provides an implicit regularising effect by concentrating gradient signal on the harder minority class.
* The pos_weight shifts the model's effective decision boundary: at the standard threshold of 0.50, the model already achieves a recall of **71 %** on defaulters and an expected cost of **0.5747** — matching the cost-sensitive SVM (0.5736) and surpassing it in ROC-AUC (0.7678 vs 0.7576). Lowering the threshold further to 0.20 raises expected cost to **0.6474**, because the model has already internalised the cost asymmetry and additional threshold reduction only accumulates false positives. This mirrors the finding from Section 2.4: when cost weighting is embedded in the learner, threshold tuning at t = 0.20 provides no further benefit and can be counterproductive.
* The MLP's performance on this moderately-sized tabular dataset (21 000 rows, 24 features) is competitive but does not surpass gradient boosting. This is consistent with the broader machine learning literature, where tree-based methods tend to dominate on structured tabular data while neural networks excel on unstructured data such as images, text, and time series.

---

## 4.6 Explainability with SHAP

Neural networks are often criticised as black boxes. **SHAP** (SHapley Additive exPlanations) provides a principled, game-theoretic attribution of each feature's contribution to a specific prediction. For PyTorch models, `shap.GradientExplainer` uses **integrated gradients**: it linearly interpolates each input from a background (reference) value to the actual value and accumulates gradients along the path, producing an attribution that satisfies the *efficiency* and *symmetry* axioms of Shapley values.

::: {#e3538561 .cell execution_count=45}
``` {.python .cell-code}
import shap

mlp_final.eval()

# GradientExplainer requires the model to return a 2-D tensor (B, n_outputs).
# Our forward() squeezes to (B,), so we wrap it to restore the trailing dim.
class _SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).unsqueeze(1)   # (B,) → (B, 1)

shap_model = _SHAPWrapper(mlp_final).to(device)
shap_model.eval()

# Background: 500-row training sample (reference distribution)
bg_tensor = torch.tensor(
    X_train_scaled.sample(500, random_state=42).values,
    dtype=torch.float32
).to(device)

# Test observations to explain
n_explain   = 1000
X_exp_np    = X_test_scaled.values[:n_explain]
X_exp_t     = torch.tensor(X_exp_np, dtype=torch.float32).to(device)

explainer    = shap.GradientExplainer(shap_model, bg_tensor)
shap_values  = explainer.shap_values(X_exp_t)

# GradientExplainer returns a list with one array per output
if isinstance(shap_values, list):
    shap_arr = shap_values[0]
else:
    shap_arr = shap_values
shap_arr = np.squeeze(shap_arr)   # ensure (n_explain, n_features) — drops trailing dim from wrapper

print(f"SHAP values shape: {shap_arr.shape}")
print(f"Explained observations: {n_explain} test samples")
```

::: {.cell-output .cell-output-stdout}
```
SHAP values shape: (1000, 24)
Explained observations: 1000 test samples
```
:::
:::


::: {#1e3425ce .cell execution_count=46}
``` {.python .cell-code}
# Beeswarm summary plot: SHAP value vs feature value for every observation
shap.summary_plot(
    shap_arr,
    features=X_exp_np,
    feature_names=X_test_scaled.columns.tolist(),
    plot_type='dot',
    show=False,
    max_display=23,
)
plt.title('SHAP Summary Plot — Tuned MLP (Credit Default)')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-47-output-1.png){width=739 height=1017}
:::
:::


::: {#e1d25efa .cell execution_count=47}
``` {.python .cell-code}
# Global feature importance: mean |SHAP| per feature
shap.summary_plot(
    shap_arr,
    features=X_exp_np,
    feature_names=X_test_scaled.columns.tolist(),
    plot_type='bar',
    show=False,
    max_display=23,
)
plt.title('Mean |SHAP| — Global Feature Importance (MLP)')
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-48-output-1.png){width=758 height=1018}
:::
:::


::: {#df85ee4f .cell execution_count=48}
``` {.python .cell-code}
# Cross-check: MLP SHAP importance vs Random Forest feature importance
mean_shap = np.abs(shap_arr).mean(axis=0)
shap_rank = pd.DataFrame({
    'Feature':   X_test_scaled.columns,
    'Mean |SHAP|': mean_shap,
    'RF Gini':   rf_best.feature_importances_,
}).sort_values('Mean |SHAP|', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(shap_rank['Feature'][::-1], shap_rank['Mean |SHAP|'][::-1],
             color='steelblue')
axes[0].set_title('MLP: Mean |SHAP| ranking')
axes[0].set_xlabel('Mean |SHAP value|')

shap_rank_rf = shap_rank.sort_values('RF Gini', ascending=False)
axes[1].barh(shap_rank_rf['Feature'][::-1], shap_rank_rf['RF Gini'][::-1],
             color='tab:orange')
axes[1].set_title('Random Forest: Gini Importance ranking')
axes[1].set_xlabel('Mean Decrease in Gini')

plt.suptitle('Feature Importance: MLP (SHAP) vs Random Forest (Gini)', y=1.01)
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-49-output-1.png){width=1334 height=585}
:::
:::


### Interpretation

* The **SHAP beeswarm plot** shows all 24 features and confirms that the MLP's top drivers are the repayment-status variables (`status_sep`, `status_aug`, etc.), matching the Random Forest's Gini importance. High positive SHAP values (pushing the default logit upward) are associated with high feature values, i.e. more months of delayed payment, confirming that the MLP has learned an economically intuitive relationship.
* The **mean |SHAP| bar chart** provides a global ranking: the six repayment-status features occupy the top positions, followed by bill and payment amounts. Demographic variables (`gender`, `marital_status`, `education`) have near-zero mean SHAP values, consistent with their negligible Gini importance in the Random Forest.
* The **side-by-side comparison** between MLP SHAP ranking and RF Gini importance shows strong agreement at the top but some divergence in the middle ranks, reflecting the non-linear combinations the network can learn across bill and payment features. Despite this, the overall feature hierarchy is consistent across model families, reinforcing the conclusion that **payment history is the dominant predictor of default** in this dataset.
* SHAP attributions are additive: the sum of SHAP values for any observation equals the difference between the model output and the expected output over the background set. This property makes SHAP explanations faithful to the model, unlike post-hoc heuristics such as permutation importance, which can be misleading when features are correlated.

---

# 5. Overall Model Comparison

## 5.1 Summary Table

The table below collects every model configuration from both Part 1 and Part 2, evaluated on the same held-out test set. All probabilistic models are shown at both the standard threshold (0.50) and the cost-optimal threshold (0.20).

::: {#84044174 .cell execution_count=49}
``` {.python .cell-code}
final_df = pd.DataFrame(all_results).set_index('Model').round(4)
display(final_df.sort_values('ROC-AUC', ascending=False))
```

::: {.cell-output .cell-output-display}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Recall (Default)</th>
      <th>Precision (Default)</th>
      <th>F1 (Default)</th>
      <th>ROC-AUC</th>
      <th>Expected Cost</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>XGBoost (t=0.2)</th>
      <td>0.7369</td>
      <td>0.6620</td>
      <td>0.4374</td>
      <td>0.5268</td>
      <td>0.7809</td>
      <td>0.5622</td>
    </tr>
    <tr>
      <th>XGBoost</th>
      <td>0.8177</td>
      <td>0.3551</td>
      <td>0.6645</td>
      <td>0.4628</td>
      <td>0.7809</td>
      <td>0.7530</td>
    </tr>
    <tr>
      <th>GBM (t=0.2)</th>
      <td>0.7396</td>
      <td>0.6610</td>
      <td>0.4409</td>
      <td>0.5289</td>
      <td>0.7809</td>
      <td>0.5604</td>
    </tr>
    <tr>
      <th>GBM</th>
      <td>0.8173</td>
      <td>0.3561</td>
      <td>0.6620</td>
      <td>0.4631</td>
      <td>0.7809</td>
      <td>0.7524</td>
    </tr>
    <tr>
      <th>MLP</th>
      <td>0.6844</td>
      <td>0.7072</td>
      <td>0.3842</td>
      <td>0.4979</td>
      <td>0.7678</td>
      <td>0.5747</td>
    </tr>
    <tr>
      <th>MLP (t=0.2)</th>
      <td>0.3979</td>
      <td>0.9488</td>
      <td>0.2621</td>
      <td>0.4108</td>
      <td>0.7678</td>
      <td>0.6474</td>
    </tr>
    <tr>
      <th>Random Forest (t=0.2)</th>
      <td>0.6853</td>
      <td>0.6906</td>
      <td>0.3829</td>
      <td>0.4927</td>
      <td>0.7599</td>
      <td>0.5884</td>
    </tr>
    <tr>
      <th>Random Forest</th>
      <td>0.8147</td>
      <td>0.3666</td>
      <td>0.6420</td>
      <td>0.4668</td>
      <td>0.7599</td>
      <td>0.7458</td>
    </tr>
    <tr>
      <th>SVM RBF (class_weight)</th>
      <td>0.6771</td>
      <td>0.7167</td>
      <td>0.3786</td>
      <td>0.4955</td>
      <td>0.7576</td>
      <td>0.5736</td>
    </tr>
    <tr>
      <th>SVM RBF (class_weight) (t=0.2)</th>
      <td>0.6870</td>
      <td>0.7022</td>
      <td>0.3860</td>
      <td>0.4981</td>
      <td>0.7576</td>
      <td>0.5766</td>
    </tr>
    <tr>
      <th>k-NN (k=59)</th>
      <td>0.8061</td>
      <td>0.2762</td>
      <td>0.6440</td>
      <td>0.3866</td>
      <td>0.7575</td>
      <td>0.8343</td>
    </tr>
    <tr>
      <th>Decision Tree</th>
      <td>0.8120</td>
      <td>0.3425</td>
      <td>0.6404</td>
      <td>0.4463</td>
      <td>0.7568</td>
      <td>0.7698</td>
    </tr>
    <tr>
      <th>Decision Tree (t=0.2)</th>
      <td>0.7072</td>
      <td>0.6549</td>
      <td>0.4010</td>
      <td>0.4974</td>
      <td>0.7568</td>
      <td>0.5981</td>
    </tr>
    <tr>
      <th>Bagging (t=0.2)</th>
      <td>0.6661</td>
      <td>0.7077</td>
      <td>0.3677</td>
      <td>0.4839</td>
      <td>0.7544</td>
      <td>0.5926</td>
    </tr>
    <tr>
      <th>Bagging</th>
      <td>0.8158</td>
      <td>0.3757</td>
      <td>0.6432</td>
      <td>0.4743</td>
      <td>0.7544</td>
      <td>0.7367</td>
    </tr>
    <tr>
      <th>LDA</th>
      <td>0.8066</td>
      <td>0.2697</td>
      <td>0.6517</td>
      <td>0.3815</td>
      <td>0.7401</td>
      <td>0.8397</td>
    </tr>
    <tr>
      <th>LDA (t=0.2)</th>
      <td>0.7250</td>
      <td>0.6339</td>
      <td>0.4195</td>
      <td>0.5049</td>
      <td>0.7401</td>
      <td>0.5990</td>
    </tr>
    <tr>
      <th>GNB</th>
      <td>0.7637</td>
      <td>0.5113</td>
      <td>0.4687</td>
      <td>0.4891</td>
      <td>0.7396</td>
      <td>0.6688</td>
    </tr>
    <tr>
      <th>SVM Linear (t=0.2)</th>
      <td>0.7031</td>
      <td>0.6504</td>
      <td>0.3959</td>
      <td>0.4922</td>
      <td>0.7394</td>
      <td>0.6062</td>
    </tr>
    <tr>
      <th>SVM Linear</th>
      <td>0.8042</td>
      <td>0.2446</td>
      <td>0.6537</td>
      <td>0.3560</td>
      <td>0.7394</td>
      <td>0.8642</td>
    </tr>
    <tr>
      <th>SVM RBF (t=0.2)</th>
      <td>0.8030</td>
      <td>0.4380</td>
      <td>0.5714</td>
      <td>0.4959</td>
      <td>0.7082</td>
      <td>0.6943</td>
    </tr>
    <tr>
      <th>SVM RBF</th>
      <td>0.8094</td>
      <td>0.2853</td>
      <td>0.6605</td>
      <td>0.3985</td>
      <td>0.7082</td>
      <td>0.8230</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::


## 5.2 Combined ROC Curve

::: {#a7189c7a .cell execution_count=50}
``` {.python .cell-code}
fig, ax = plt.subplots(figsize=(9, 7))

colours = {
    'LDA':                    '#1f77b4',
    'GNB':                    '#ff7f0e',
    'k-NN (k=59)':            '#2ca02c',
    'SVM Linear':             '#d62728',
    'SVM RBF':                '#9467bd',
    'SVM RBF (class_weight)': '#c5b0d5',
    'Decision Tree':          '#8c564b',
    'Bagging':                '#e377c2',
    'Random Forest':          '#7f7f7f',
    'GBM':                    '#bcbd22',
    'XGBoost':                '#17becf',
    'MLP':                    '#aec7e8',
}

for name, prob in all_probs.items():
    auc_val = roc_auc_score(y_test, prob)
    RocCurveDisplay.from_predictions(
        y_test, prob, ax=ax,
        name=f'{name} (AUC = {auc_val:.3f})',
        color=colours.get(name, None)
    )

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
ax.set_title('ROC Curves: All Models (Parts 1 & 2)')
ax.legend(loc='lower right', fontsize=7.5)
plt.tight_layout()
plt.show()
```

::: {.cell-output .cell-output-display}
![](second_project_files/figure-html/cell-51-output-1.png){width=654 height=660}
:::
:::


### Interpretation

* The full comparison table reveals a clear **ranking by ROC-AUC**: GBM and XGBoost lead at **0.7809**, followed by MLP (0.7678), Random Forest (0.7599), cost-sensitive SVM (0.7576), k-NN (0.7575), Decision Tree (0.7568), Bagging (0.7544), LDA (0.7401), GNB (0.7396), Linear SVM (0.7394), and RBF SVM (0.7082).
* When evaluated by **best achieved expected cost** (the metric most relevant to the bank's decision problem), the ranking shifts: **GBM (t = 0.20)** achieves the lowest expected cost at **0.5604**, followed by XGBoost (0.5622), cost-sensitive SVM (0.5736, at t = 0.50), MLP (0.5747, at t = 0.50), and Random Forest (0.5884, at t = 0.20). All five improve on Part 1's best result (LDA at t = 0.20, expected cost 0.5990). For GBM, XGBoost, and Random Forest the best cost is achieved by lowering the threshold to 0.20; for the SVM with class_weight and the MLP with pos_weight, cost is lowest at the standard threshold because the cost asymmetry is already embedded in the learner.
* The ROC overlay shows that the GBM/XGBoost curves clearly dominate the upper-left region, separating themselves from the rest of the field. The clustering of LDA, GNB, and Linear SVM in the lower ROC band confirms that these linear methods have reached their ceiling on this dataset.
* A consistent pattern holds for models trained without cost weighting: at the standard threshold (t = 0.50), they achieve accuracy ≈ 80–82 % but very low recall on defaulters (24–39 %). Lowering the threshold to 0.20 raises recall to 63–72 % at the cost of accuracy, reducing expected cost by roughly 20–30 %. For these models, **threshold calibration is the single most impactful intervention**. Cost-weighted models (SVM with class_weight, MLP with pos_weight = 5) break this pattern: they already achieve high recall (≈ 70 %) and competitive expected cost at the standard threshold, and further lowering to 0.20 offers no benefit, remaining neutral or slightly increasing cost.
* Both cost-weighted models behave this way: the SVM with class_weight achieves expected cost **0.5736** at t = 0.50, and the MLP with pos_weight = 5 achieves **0.5747** at t = 0.50. In both cases the cost asymmetry is incorporated into the training objective, so the learner's outputs already reflect the 5:1 penalty without requiring threshold adjustment.

---

# 6. Conclusion

Across both parts of this project, we have evaluated eleven modelling approaches on the UCI Credit Card Default dataset (30 000 cardholders, 22 % default prevalence). The results can be distilled into several key findings:

**1. Gradient boosting methods (GBM, XGBoost) deliver the best overall performance.** Both achieve **ROC-AUC = 0.7809** and the lowest expected costs at the cost-optimal threshold (**0.5604 and 0.5622**, respectively). Their sequential, residual-fitting strategy is well-suited to this tabular dataset, where the signal is distributed across multiple repayment-history features that interact in non-trivial ways.

**2. The MLP neural network is competitive but does not surpass tree-based ensembles on this structured data.** The tuned MLP achieves ROC-AUC = 0.7678, placing it between Random Forest and GBM. This outcome aligns with the broader empirical finding that gradient boosting tends to outperform deep learning on moderate-sized tabular datasets. The MLP's tendency to overfit (training AUC 0.83 vs validation AUC 0.78 after 40 epochs) suggests that additional regularisation (e.g. early stopping, weight decay) or a larger dataset would be needed to close the gap.

**3. Threshold calibration is more impactful than model selection for most models.** Moving from the default threshold (0.50) to the cost-optimal threshold (0.20) reduces expected cost by roughly 20–30 % for models trained without cost weighting. For cost-weighted models (SVM with class_weight, MLP with pos_weight), the threshold reduction offers no benefit. For a bank, this means that threshold calibration is highly effective, but its impact depends on whether cost information has already been incorporated into training.

**4. Cost-sensitive training is an effective alternative to threshold tuning.** Both the SVM with class_weight={0:1, 1:5} (expected cost 0.5736 at t = 0.50) and the MLP with pos_weight = 5 (expected cost 0.5747 at t = 0.50) achieve competitive expected costs comparable to GBM at its cost-optimal threshold, because the cost asymmetry is incorporated directly into the training objective rather than applied post-hoc.

**5. Feature importance is consistent across model families.** Both the Random Forest's Gini importance and the MLP's SHAP attributions identify the same top features: the six monthly repayment-status variables (`status_sep` through `status_apr`). Demographic variables (gender, marital status, education) contribute almost nothing. This consistency across fundamentally different model architectures strengthens our confidence that **payment behaviour, not demographics, drives default risk** in this dataset.

**6. Linear methods (LDA, Linear SVM) provide a solid baseline but have a low ceiling.** Both achieve ROC-AUC ≈ 0.74, confirming that a linear boundary captures a substantial portion of the discriminative signal. However, they cannot match the non-linear methods' ability to exploit feature interactions, leaving approximately 4 percentage points of AUC on the table.

In summary, for a credit card issuer deploying a default prediction model on a population similar to this dataset, we recommend **GBM or XGBoost with a cost-calibrated threshold** as the primary production model, complemented by a **simple decision tree or logistic model for interpretability and regulatory explanation**. The consistency of feature importance rankings across all model families provides additional assurance that the models are capturing genuine economic signal rather than artefacts of a particular algorithm.

