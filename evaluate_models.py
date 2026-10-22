"""
Model-evaluation toolkit: stratified k-fold CV, confusion matrix, ROC-AUC,
precision-recall curves, and a demonstration of why accuracy misleads on
imbalanced data.

Run:  python evaluate_models.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 5
RESULTS = Path("results")


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def load_pima(path="diabetes.csv"):
    """Pima Indians Diabetes (~35% positive). Zeros in some columns mean 'missing'."""
    df = pd.read_csv(path)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
    return df.drop(columns="Outcome"), df["Outcome"]


def load_imbalanced(n=10_000, minority=0.03):
    """Synthetic, heavily imbalanced data (~3% positive), fraud-detection style."""
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        n_informative=6,
        n_redundant=2,
        weights=[1 - minority, minority],
        class_sep=0.8,
        flip_y=0.01,
        random_state=SEED,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), pd.Series(y)


# --------------------------------------------------------------------------
# Models: every preprocessing step lives INSIDE the pipeline, so during CV the
# imputer/scaler are re-fit on each training fold only -> no data leakage.
# --------------------------------------------------------------------------
def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ]),
    }


# --------------------------------------------------------------------------
# Evaluation pieces
# --------------------------------------------------------------------------
def run_cross_validation(models, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            scores = res[f"test_{metric}"]
            row[metric] = f"{scores.mean():.3f} +/- {scores.std():.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_confusion_matrices(fitted, X_te, y_te, out):
    fig, axes = plt.subplots(1, len(fitted), figsize=(5 * len(fitted), 4))
    for ax, (name, model) in zip(np.atleast_1d(axes), fitted.items()):
        ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc_pr(fitted, X_te, y_te, out_roc, out_pr):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, name=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_roc, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        ap = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
        PrecisionRecallDisplay.from_estimator(model, X_te, y_te, ax=ax, name=f"{name} (AP={ap:.3f})")
    ax.axhline(y_te.mean(), color="k", linestyle="--", label=f"Baseline (prevalence={y_te.mean():.3f})")
    ax.set_title("Precision-recall curves (test set)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_pr, dpi=150)
    plt.close(fig)


def accuracy_trap(X_tr, y_tr, X_te, y_te):
    """A model that always predicts the majority class: high accuracy, zero value."""
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    pred = dummy.predict(X_te)
    return {
        "accuracy": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred, zero_division=0),
        "recall": float((pred[y_te == 1] == 1).mean()),
    }


def evaluate_dataset(tag, X, y):
    print(f"\n{'=' * 70}\nDATASET: {tag}  (n={len(y)}, positive rate={y.mean():.3f})\n{'=' * 70}")
    out_dir = RESULTS / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    models = build_models()

    # 1) Stratified k-fold CV on the whole dataset: mean +/- std
    cv_table = run_cross_validation(models, X, y)
    print(f"\n{N_SPLITS}-fold stratified CV (mean +/- std):\n{cv_table.to_string()}")
    cv_table.to_csv(out_dir / "cv_results.csv")

    # 2) Held-out test split for curves / confusion matrices
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    fitted = {name: m.fit(X_tr, y_tr) for name, m in models.items()}

    for name, m in fitted.items():
        print(f"\n--- {name}: classification report (threshold 0.5) ---")
        print(classification_report(y_te, m.predict(X_te), digits=3, zero_division=0))

    plot_confusion_matrices(fitted, X_te, y_te, out_dir / "confusion_matrices.png")
    plot_roc_pr(fitted, X_te, y_te, out_dir / "roc_curves.png", out_dir / "pr_curves.png")

    # 3) Accuracy trap
    trap = accuracy_trap(X_tr, y_tr, X_te, y_te)
    print(
        "\nAccuracy trap - 'always predict majority class' baseline: "
        f"accuracy={trap['accuracy']:.3f}, F1={trap['f1']:.3f}, recall={trap['recall']:.3f}"
    )
    pd.DataFrame([trap]).to_csv(out_dir / "majority_baseline.csv", index=False)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    X, y = load_pima("diabetes.csv")
    evaluate_dataset("pima_diabetes", X, y)

    X, y = load_imbalanced()
    evaluate_dataset("imbalanced_synthetic", X, y)

    print(f"\nFigures and tables saved under: {RESULTS.resolve()}")
