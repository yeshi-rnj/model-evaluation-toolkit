# Model Evaluation Toolkit

A reusable evaluation suite for binary classifiers: **stratified k-fold cross-validation, confusion matrices, ROC-AUC, and precision-recall curves**, applied to two models on two datasets, one of them heavily imbalanced.

Follow-up to [supervised-learning-comparison](https://github.com/yeshi-rnj/supervised-learning-comparison): that repo compared 5 algorithms on a single split; this one asks *how much should we trust the numbers*, and *which metric should drive the decision*.

## What's inside

| Requirement | Where |
|---|---|
| Stratified 5-fold CV, mean +/- std | `run_cross_validation()` -> `results/<dataset>/cv_results.csv` |
| Confusion matrices (held-out test set) | `results/<dataset>/confusion_matrices.png` |
| ROC curves + AUC (both models, same axes) | `results/<dataset>/roc_curves.png` |
| Precision-recall curves + baseline | `results/<dataset>/pr_curves.png` |
| Accuracy-is-misleading demo | `accuracy_trap()` -> `results/<dataset>/majority_baseline.csv` |

## Datasets

1. **Pima Indians Diabetes** (`diabetes.csv`): 768 rows, ~35% positive. Mildly imbalanced, real data.
2. **Synthetic imbalanced set** (`make_classification`): 10,000 rows, ~3% positive, fraud-detection style. Fixed seed, so it's reproducible.

## Models

- Logistic Regression (median-impute -> scale -> LR)
- Random Forest (median-impute -> RF, 300 trees)

## No data leakage

Imputation and scaling live **inside** an sklearn `Pipeline`, so during cross-validation they are re-fit on each training fold only. Folds are `StratifiedKFold(shuffle=True, random_state=42)` so every fold keeps the class ratio. The test split is stratified and never used for fitting or tuning.

## Run it (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python evaluate_models.py
```

## Results

> Paste your own numbers from `results/` after running the script.

### Pima Diabetes (5-fold CV)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |

### Synthetic imbalanced (5-fold CV)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |

**The accuracy trap:** a model that always predicts "negative" scores `___` accuracy on the imbalanced set, with recall `0.0` and F1 `0.0`. Accuracy alone would call it excellent while it catches zero positives.

## When is PR more informative than ROC?

ROC plots TPR vs FPR. FPR divides false positives by the (huge) number of negatives, so with a rare positive class even many false alarms barely move the FPR and ROC-AUC stays flattering. Precision divides false positives by the number of *predicted positives*, so it falls fast when false alarms swamp true hits. The PR curve (and its baseline, the positive rate) therefore exposes weaknesses ROC hides. Use PR when the positive class is rare and you care about finding it; ROC is fine when classes are roughly balanced.

## Which model would I ship?

> Write this yourself from your results. Cover: (1) which model wins on CV mean +/- std and whether the gap is bigger than the std; (2) what the confusion matrix says about the costs of false negatives vs false positives for this problem; (3) PR-AUC vs ROC-AUC; (4) whether you'd tune the decision threshold; (5) one limitation.

## Repo structure

```
.
├── evaluate_models.py
├── diabetes.csv          # copy from supervised-learning-comparison
├── requirements.txt
├── results/              # generated
└── README.md
```
