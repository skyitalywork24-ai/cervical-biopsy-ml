import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, classification_report, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

st.set_page_config(page_title="Cervical Biopsy Risk Lab", page_icon="🩺", layout="wide")

TARGET = "Biopsy"
MODEL_DIR = Path("model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine (RBF)": "svm.joblib",
}


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


@st.cache_data
def load_features():
    return json.load(open(MODEL_DIR / "feature_columns.json"))


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------- Sidebar ----------------
st.sidebar.title("🩺 Biopsy Risk Lab")
st.sidebar.caption("Compare five classifiers on cervical cancer risk-factor data.")
uploaded = st.sidebar.file_uploader("Upload test CSV (features + Biopsy)", type="csv")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=uploaded is None)
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Probability above which a patient is flagged as biopsy-positive.")

st.title("Cervical Cancer Biopsy Prediction")
st.write("Upload a held-out test set. Models are pre-trained and are **not** retrained on your upload.")

# ---------------- Load data ----------------
if uploaded is not None:
    df = pd.read_csv(uploaded, na_values="?")
elif use_sample and Path("test_data.csv").exists():
    df = pd.read_csv("test_data.csv", na_values="?")
else:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
features = load_features()
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(df))
c2.metric("Features", len(features))
c3.metric("Biopsy-positive", f"{y.sum()} ({y.mean():.1%})")
with st.expander("Preview data"):
    st.dataframe(df.head(20), width="stretch")

tab_single, tab_compare = st.tabs(["Selected model", "Compare all models"])

# ---------------- Single model ----------------
with tab_single:
    model = load_model(MODEL_FILES[model_name])
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    m = compute_metrics(y, pred, prob)

    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y, pred, display_labels=["Negative", "Positive"], cmap="Purples", ax=ax)
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        RocCurveDisplay.from_predictions(y, prob, ax=ax)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        st.pyplot(fig)

    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=["Negative", "Positive"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ---------------- Compare ----------------
with tab_compare:
    rows = {}
    for name, f in MODEL_FILES.items():
        mdl = load_model(f)
        p = mdl.predict_proba(X)[:, 1]
        rows[name] = compute_metrics(y, (p >= threshold).astype(int), p)
    table = pd.DataFrame(rows).T.round(4)
    st.subheader("All models on the uploaded data")
    st.dataframe(table.style.highlight_max(axis=0, color="#d9c8f5"), width="stretch")

    fig, ax = plt.subplots(figsize=(9, 4))
    table.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticklabels(table.index, rotation=20, ha="right")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=8)
    st.pyplot(fig)
    st.caption("With imbalanced data, watch Recall, F1 and MCC rather than Accuracy alone.")
