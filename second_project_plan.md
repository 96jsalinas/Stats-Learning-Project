# Second Project Plan: Machine Learning Tools for Credit Card Default

---

## 1. Context & Carry-Over from Project 1

### Dataset
UCI Default of Credit Card Clients — 30 000 Taiwanese cardholders, 2005.
Binary target: `default` (1 = defaulted on October 2005 payment, 22 % prevalence).

### Style conventions to match
- Quarto `.qmd` format, `jupyter: python3`, `code-fold: true`, `toc: true`.
- Numbered top-level sections (`# 1.`, `# 2.`, …), subsections (`## 2.1`, `### Theory`, `### Fit & Evaluate`, `### Interpretation`).
- Each subsection: short prose intro → code cell → bullet-point interpretation block.
- Metrics reported to **4 decimal places**; DataFrames shown via `display()`.
- Plots: `matplotlib` + `seaborn`, `plt.tight_layout()`, consistent palette (`viridis`, `Set2`, `tab:*`).
- Avoid accuracy as the sole metric — always report **ROC-AUC, Recall (Default), F1 (Default), Expected Cost** alongside accuracy.

### Preprocessing (reuse exactly from Project 1)
```python
# Already produced in Project 1 — carry over these objects:
X_train_scaled, X_test_scaled   # pd.DataFrame, StandardScaler applied
y_train, y_test                 # pd.Series, stratified 70/30 split
# Cost matrix (FN=5, FP=1) and helper
C = np.array([[0, 1], [5, 0]])
opt_t = 0.200   # cost-optimal threshold from Section 7
```
We will **not** repeat the EDA or preprocessing sections. The second report opens with a brief recap table and immediately proceeds to modelling.

---

## 2. Techniques to Implement

The source notebooks define four families of methods. We map each to the credit-default problem:

| Source file | Technique | Our task |
|---|---|---|
| `topic-6-svm.Rmd` | Linear SVM, Soft-margin SVM, RBF/Poly kernel, Grid search C×γ | Binary classification of default |
| `topic-7-trees-forests.Rmd` | Decision tree + pruning, Bagging, Random Forest, GBM, XGBoost | Binary classification of default |
| `topic-8-neural-networks.ipynb` | MLP (PyTorch), training loop, hyperparameter search, learning curves | Binary classification of default |
| `topic-5-knn.Rmd` | k-NN (distance-based, k tuning) | **Already covered in Project 1, Section 8 — skip** |

---

## 3. Document Structure

```
# Overview (brief recap of dataset, preprocessing, evaluation framework)

# 1. Support Vector Machines
## 1.1 Theory
## 1.2 Linear SVM (Baseline)
## 1.3 Kernel SVM (RBF) with Hyperparameter Tuning
### Fit & Evaluate
### Interpretation
## 1.4 Cost-Sensitive SVM (class_weight & threshold tuning)
## 1.5 Summary

# 2. Tree-Based Methods
## 2.1 Single Decision Tree
### Theory
### Fit & Evaluate (with cost-complexity pruning)
### Interpretation
## 2.2 Bagging
## 2.3 Random Forest
### Theory
### mtry (max_features) Tuning via OOB Error
### Variable Importance
### Interpretation
## 2.4 Gradient Boosting (GBM)
## 2.5 XGBoost
## 2.6 Summary Table

# 3. Neural Networks (MLP)
## 3.1 Theory
## 3.2 Architecture & Training Loop
## 3.3 Hyperparameter Search
## 3.4 Final Evaluation & Learning Curves
## 3.5 Interpretation

# 4. Overall Model Comparison
## 4.1 Summary Table (all models, both projects)
## 4.2 ROC Curve Overlay
## 4.3 Key Takeaways

# 5. Conclusion
```

---

## 4. Section-by-Section Implementation Plan

### Section 1 — Support Vector Machines

**Theory prose** (matching source `topic-6-svm.Rmd` progression):
- Hard-margin MMC → Soft-margin SVC (cost C) → Kernel SVM (kernel trick, RBF formula).
- Note that scikit-learn's `C` behaves inversely to the slack budget: large C = tight margin.

**1.2 Linear SVM baseline**
```python
from sklearn.svm import SVC
svm_lin = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_lin.fit(X_train_scaled, y_train)
# → classification_report, roc_auc_score, confusion matrix, ROC curve
```

**1.3 Kernel SVM — RBF with GridSearchCV**
```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gs = GridSearchCV(svm_rbf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
gs.fit(X_train_scaled, y_train)
# → heatmap of CV AUC (C × gamma), best params, test evaluation
```
Visualise the grid-search results as a heatmap (C on y-axis, gamma on x-axis, cell colour = CV AUC) — mirrors the R source.

**1.4 Cost-sensitive SVM**
- Refit best kernel SVM with `class_weight={0: 1, 1: 5}` (matching the 5:1 FN/FP cost matrix).
- Apply cost-optimal threshold 0.200 to probabilities.
- Compare expected cost across variants.

**1.5 Summary** — comparison table: Linear SVM / RBF default / RBF balanced weights / RBF cost threshold.

**Plot types:** heatmap (grid search), confusion matrices (side-by-side), ROC overlay.

---

### Section 2 — Tree-Based Methods

**2.1 Single Decision Tree**
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_val_score

# Grow a full tree, then prune via cost-complexity path
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train_scaled, y_train)

path = dt_full.cost_complexity_pruning_path(X_train_scaled, y_train)
# Cross-validate over alpha values → pick alpha with min CV error
# (mirrors R's cv.tree + prune.misclass)
alphas = path.ccp_alphas
cv_scores = [cross_val_score(
    DecisionTreeClassifier(ccp_alpha=a, random_state=42),
    X_train_scaled, y_train, cv=5, scoring='roc_auc'
).mean() for a in alphas]
best_alpha = alphas[np.argmax(cv_scores)]

dt_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
dt_pruned.fit(X_train_scaled, y_train)
```
Plot: CV AUC vs alpha (pruning curve), tree diagram with `plot_tree` (depth-limited for readability).

**2.2 Bagging**
```python
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=300, max_features=1.0,   # all features = bagging
    oob_score=True, random_state=42, n_jobs=-1
)
bag.fit(X_train_scaled, y_train)
print(f"OOB score: {bag.oob_score_:.4f}")
```

**2.3 Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

# Tune max_features (equivalent to mtry in R)
# Default sqrt(p) for classification
p = X_train_scaled.shape[1]
mtry_grid = [max(1, int(np.sqrt(p)) - 1), int(np.sqrt(p)), int(np.sqrt(p)) + 1, p // 3, p]

oob_results = []
for m in mtry_grid:
    rf = RandomForestClassifier(n_estimators=300, max_features=m,
                                 oob_score=True, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    oob_results.append({'max_features': m, 'OOB_error': 1 - rf.oob_score_})

best_m = min(oob_results, key=lambda x: x['OOB_error'])['max_features']
rf_best = RandomForestClassifier(n_estimators=350, max_features=best_m,
                                  oob_score=True, random_state=42, n_jobs=-1)
rf_best.fit(X_train_scaled, y_train)
```
Plots: OOB error vs n_estimators (convergence curve), OOB error vs max_features, horizontal bar chart of `feature_importances_` (MeanDecreaseGini equivalent).

**2.4 GBM**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

gbm_grid = {
    'n_estimators': [100, 250],
    'max_depth': [2, 3],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8]
}
gbm = GradientBoostingClassifier(random_state=42)
gs_gbm = GridSearchCV(gbm, gbm_grid, cv=3, scoring='roc_auc', n_jobs=-1)
gs_gbm.fit(X_train_scaled, y_train)
```

**2.5 XGBoost**
```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective='binary:logistic', eval_metric='auc',
    eta=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8,
    n_estimators=250, random_state=42, use_label_encoder=False
)
xgb.fit(X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)], verbose=False)
# plot xgb.evals_result() learning curve
```

**2.6 Summary Table** — identical columns to Project 1 comparison tables.

---

### Section 3 — Neural Networks (MLP with PyTorch)

Architecture mirrors `topic-8-neural-networks.ipynb` but adapted for tabular binary classification:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class CreditDefaultMLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),  nn.ReLU(),
            nn.Linear(hidden2, 1)         # single logit → BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x).squeeze(1)
```

**Loss:** `nn.BCEWithLogitsLoss(pos_weight=tensor([4.0]))` — encodes the 5:1 FN/FP cost asymmetry directly (pos_weight ≈ n_neg/n_pos × cost_ratio adjustment).

**Training loop:** reuse the pattern from the notebook (`train_one_epoch`, `evaluate`, `fit_model`) adapted for single-output binary classification.

**Plots:** loss + AUC learning curves (train vs validation, by epoch), confusion matrix at optimal threshold.

**Hyperparameter search** (manual grid, short epochs per config, full retrain of winner):
```python
grid = [
    {'hidden1': 128, 'hidden2': 64,  'dropout': 0.2, 'lr': 1e-3},
    {'hidden1': 256, 'hidden2': 128, 'dropout': 0.2, 'lr': 1e-3},
    {'hidden1': 128, 'hidden2': 64,  'dropout': 0.3, 'lr': 5e-4},
    {'hidden1': 256, 'hidden2': 128, 'dropout': 0.3, 'lr': 5e-4},
]
# Rank by best validation AUC after N_quick epochs on a 60% subset
# Retrain winner on full training set for N_full epochs
```

**Explainability with SHAP (added):**
- Library: `shap` — `pip install shap`.
- Explainer: `shap.GradientExplainer(model, background)` — uses integrated gradients, native PyTorch support, more robust than `DeepExplainer` for arbitrary architectures.
- Background: 500-row training sample (reference expectation); explained set: 1 000 test observations.
- `GradientExplainer.shap_values()` returns a list with one element for a single-output model; extract `shap_values[0]` → shape `(1000, 23)`.
- Plots: (1) beeswarm summary — each dot is one observation coloured by feature value, x-axis = SHAP value for the default logit; (2) horizontal bar of mean |SHAP| for a clean global feature ranking comparable to tree-based variable importance.
- This lets us cross-check which features the MLP considers important against the Random Forest's `feature_importances_`.

---

### Section 4 — Overall Comparison

Collect every model from **both projects** (LDA, QDA, GNB, k-NN, SVM variants, tree methods, MLP) into one final DataFrame:

| Model | Accuracy | Recall (Default) | Precision (Default) | F1 (Default) | ROC-AUC | Expected Cost |
|---|---|---|---|---|---|---|
| LDA (cost threshold) | … | … | … | … | … | … |
| … | | | | | | |
| XGBoost | … | … | … | … | … | … |
| MLP (best config) | … | … | … | … | … | … |

- All non-probabilistic tree predictions evaluated at both threshold 0.50 and the cost-optimal 0.20.
- Combined ROC curve overlay plot (one curve per model family, different colours).
- Discussion: which model family wins on AUC? On expected cost? On interpretability?

---

## 5. Python Libraries

| Purpose | Library / function |
|---|---|
| Data & preprocessing | `pandas`, `numpy` (carry-over from Project 1) |
| SVM | `sklearn.svm.SVC` |
| Decision tree + pruning | `sklearn.tree.DecisionTreeClassifier`, `cost_complexity_pruning_path` |
| Bagging | `sklearn.ensemble.BaggingClassifier` |
| Random forest | `sklearn.ensemble.RandomForestClassifier` |
| GBM | `sklearn.ensemble.GradientBoostingClassifier` |
| XGBoost | `xgboost.XGBClassifier` |
| Neural network | `torch`, `torch.nn`, `torch.utils.data` (PyTorch) |
| Cross-validation & tuning | `sklearn.model_selection.GridSearchCV`, `StratifiedKFold`, `cross_val_score` |
| Metrics | `sklearn.metrics` (same as Project 1) |
| Visualisation | `matplotlib.pyplot`, `seaborn` |

No new data-loading dependencies are needed: the preprocessed arrays are carried over.

---

## 6. Key Design Decisions

1. **Single preprocessing block** at the top of the document (copy-paste from Project 1, no re-explanation). Keeps the report self-contained without repeating EDA.

2. **Threshold = 0.200 applied consistently**: every probabilistic model outputs probabilities that are evaluated both at 0.50 (standard) and 0.20 (cost-optimal) so results are comparable with Project 1.

3. **`class_weight='balanced'` or `pos_weight`** as the model-level counterpart to threshold tuning, for SVM and MLP respectively. This mirrors Project 1's Section 7 (balanced priors for LDA).

4. **OOB error instead of CV** for tree ensembles (Random Forest, Bagging): matches the R source philosophy and avoids extra computation on 21 000 training rows.

5. **XGBoost fitted on raw numpy arrays** (not scaled, trees are scale-invariant). SVM and MLP use `X_train_scaled`.

6. **MLP trained with BCEWithLogitsLoss + pos_weight**: avoids the need for a separate sigmoid; outputs a single logit per sample, converted to probability via `torch.sigmoid` at inference time.

7. **Interpretation paragraphs**: every subsection ends with 3–5 bullet points using the same pattern as Project 1 (star bold key numbers, reference earlier findings, connect to theory).

---

## 7. Estimated Section Lengths & Code Cells

| Section | Prose cells | Code cells | Plot cells |
|---|---|---|---|
| Overview / preprocessing | 1 | 1 | 0 |
| 1. SVM | 6 | 7 | 5 |
| 2. Trees | 8 | 10 | 7 |
| 3. Neural Networks | 6 | 8 | 4 |
| 4. Comparison | 2 | 2 | 2 |
| 5. Conclusion | 1 | 0 | 0 |
| **Total** | **24** | **28** | **18** |

---

## 8. Open Questions / Decisions for the Team

- **SVM runtime**: with 21 000 training rows and an RBF kernel, full GridSearchCV (4×5 grid, 3-fold CV) will be slow (~20–40 min on CPU). Options: (a) reduce to 3×4 grid, (b) use `LinearSVC` for the baseline only and limit RBF grid, (c) subsample training set to 10 000 for tuning.
- **MLP epochs**: 50–100 epochs should suffice for convergence on this tabular dataset. We can add early stopping (`patience=10` on validation AUC) to keep runtime manageable.
- **Tree visualisation**: `plot_tree` with all 23 features is unreadable. Limit `max_depth=3` in the visualisation-only call, with a note that the full model uses the pruned depth.
- **Final comparison scope**: include all models from Project 1 in Section 4, or only the best representative from each family (LDA, k-NN, SVM, RF, XGBoost, MLP)?
