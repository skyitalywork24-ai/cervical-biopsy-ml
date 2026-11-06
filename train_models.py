"""
Train five classifiers on the UCI Cervical Cancer (Risk Factors) dataset.

Usage:
    python train_models.py risk_factors_cervical_cancer.csv

Outputs:
    model/<name>.joblib      - full pipeline (imputer + scaler + classifier)
    model/feature_columns.json
    model/metrics.csv        - comparison table (paste into README)
    test_data.csv            - held-out test split (features + Biopsy) for the app
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TARGET = "Biopsy"
# The other three columns are alternative screening tests for the same
# disease. Using them as inputs would leak the answer, so they are dropped.
LEAKY = ["Hinselmann", "Schiller", "Citology"]

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
    "knn": KNeighborsClassifier(n_neighbors=7),
    "naive_bayes": GaussianNB(),
    "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
}


def load_data(path: str) -> pd.DataFrame:
    # Missing values in this dataset are written as "?"
    df = pd.read_csv(path, na_values="?")
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in LEAKY if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    return df


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main(csv_path: str):
    df = load_data(csv_path)
    print(f"Rows: {len(df)}, features: {df.shape[1] - 1}")
    print("Class balance:\n", df[TARGET].value_counts().to_string(), "\n")

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    Path("model").mkdir(exist_ok=True)
    rows = {}
    for name, clf in MODELS.items():
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows[name] = evaluate(y_test, pred, prob)
        joblib.dump(pipe, f"model/{name}.joblib")
        print(f"{name:20s}", {k: round(v, 4) for k, v in rows[name].items()})

    metrics = pd.DataFrame(rows).T.round(4)
    metrics.index.name = "Model"
    metrics.to_csv("model/metrics.csv")
    json.dump(list(X.columns), open("model/feature_columns.json", "w"))

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv("test_data.csv", index=False)
    print("\nSaved models, model/metrics.csv and test_data.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python train_models.py <path to risk_factors_cervical_cancer.csv>")
    main(sys.argv[1])
