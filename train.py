"""
Credit Card Fraud Detection - training and evaluation script
------------------------------------------------------------
What this script does (matches the synopsis methodology):
  1. Loads and cleans the transaction dataset
  2. Stratified train / validation / test split
  3. Trains Logistic Regression, Decision Tree and Random Forest
       - on the original class distribution
       - with imbalance handling (class weights, optionally SMOTE on training data only)
  4. Evaluates with accuracy, precision, recall, F1, ROC-AUC, AUC-PR, confusion matrix
  5. Threshold analysis (false positives vs false negatives)
  6. Random Forest feature importance
  7. Saves the best model (selected on the VALIDATION set) for the demo app

Usage:
  python train.py --data data/creditcard.csv      # real dataset (Kaggle MLG-ULB)
  python train.py --demo                          # quick test with synthetic data
  python train.py --data data/creditcard.csv --smote   # also try SMOTE (needs imbalanced-learn)
"""
import argparse
import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:  # optional, only needed for --smote
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "Class"
SCALE_COLS = ["Time", "Amount"]
OUT_DIR = "outputs"
MODEL_DIR = "models"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_demo_data(n=20000, fraud_rate=0.005):
    """Synthetic data with the same column layout as the Kaggle dataset."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=12, n_redundant=4,
        weights=[1 - fraud_rate, fraud_rate], class_sep=1.2, flip_y=0,
        random_state=RANDOM_STATE)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    rng = np.random.RandomState(RANDOM_STATE)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = np.round(rng.exponential(80, n), 2)
    df[TARGET] = y
    return df


def load_data(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise SystemExit(f"Column '{TARGET}' not found. Expected the creditcard.csv layout.")
    before = len(df)
    df = df.drop_duplicates().dropna()
    print(f"Rows: {before} -> {len(df)} after removing duplicates / missing values")
    return df


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_estimator(name, balanced):
    cw = "balanced" if balanced else None
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight=cw,
                                      random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2,
                                      class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE)
    raise ValueError(name)


def make_pipeline(name, scenario):
    pre = ColumnTransformer([("scale", StandardScaler(), SCALE_COLS)], remainder="passthrough")
    if scenario == "smote":
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RANDOM_STATE)),
                            ("clf", make_estimator(name, balanced=False))])
    return Pipeline([("pre", pre), ("clf", make_estimator(name, balanced=(scenario == "class_weight")))])


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics_at(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def best_threshold(y_val, proba_val):
    """Threshold with the highest F1 on the VALIDATION set (not the test set)."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (quick test)")
    ap.add_argument("--smote", action="store_true", help="also evaluate SMOTE (imbalanced-learn)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = make_demo_data() if args.demo else load_data(args.data)
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    features = list(X.columns)
    print(f"Fraud cases: {y.sum()} of {len(y)} ({100 * y.mean():.3f}%)")

    # 64% train / 16% validation / 20% test, stratified
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.20, stratify=y_tr_full, random_state=RANDOM_STATE)

    scenarios = ["original", "class_weight"]
    if args.smote:
        if HAS_IMBLEARN:
            scenarios.append("smote")
        else:
            print("imbalanced-learn not installed -> skipping SMOTE (pip install imbalanced-learn)")

    runs, rows = {}, []
    for name in MODEL_NAMES:
        for sc in scenarios:
            print(f"Training: {name} | {sc}")
            pipe = make_pipeline(name, sc).fit(X_train, y_train)
            p_val = pipe.predict_proba(X_val)[:, 1]
            p_test = pipe.predict_proba(X_test)[:, 1]
            m = metrics_at(y_test, p_test, 0.5)
            m.update({"model": name, "scenario": sc,
                      "val_pr_auc": average_precision_score(y_val, p_val)})
            rows.append(m)
            runs[(name, sc)] = (pipe, p_val, p_test)

    res = pd.DataFrame(rows)[["model", "scenario", "accuracy", "precision", "recall", "f1",
                              "roc_auc", "pr_auc", "TP", "FP", "FN", "TN", "val_pr_auc"]]
    res.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print("\n=== Test-set results at threshold 0.5 ===")
    print(res.drop(columns="val_pr_auc").round(4).to_string(index=False))

    # ---- model selection on validation AUC-PR -------------------------------
    best = res.sort_values("val_pr_auc", ascending=False).iloc[0]
    key = (best["model"], best["scenario"])
    pipe, p_val, p_test = runs[key]
    thr = best_threshold(y_val, p_val)
    final = metrics_at(y_test, p_test, thr)
    print(f"\nBest model (validation AUC-PR): {key[0]} | {key[1]}")
    print(f"Threshold chosen on validation set: {thr}")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in final.items()})

    # ---- threshold analysis (test set) --------------------------------------
    rows_t = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        m = metrics_at(y_test, p_test, t)
        rows_t.append({"threshold": t, **{k: m[k] for k in ["precision", "recall", "f1", "TP", "FP", "FN", "TN"]}})
    thr_df = pd.DataFrame(rows_t)
    thr_df.to_csv(f"{OUT_DIR}/threshold_analysis.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
    plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
    plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
    plt.axvline(thr, color="gray", linestyle="--", label=f"chosen = {thr}")
    plt.xlabel("Decision threshold"); plt.ylabel("Score"); plt.legend()
    plt.title(f"Threshold analysis - {key[0]} ({key[1]})")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/threshold_analysis.png", dpi=150); plt.close()

    # ---- precision-recall curves --------------------------------------------
    plt.figure(figsize=(7, 5))
    for (n_, s_), (_, _, pt) in runs.items():
        pr, rc, _ = precision_recall_curve(y_test, pt)
        ap_ = average_precision_score(y_test, pt)
        plt.plot(rc, pr, label=f"{n_} | {s_} (AP={ap_:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves (test set)")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pr_curves.png", dpi=150); plt.close()

    # ---- confusion matrix of the best model ---------------------------------
    cm = np.array([[final["TN"], final["FP"]], [final["FN"], final["TP"]]])
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Legit", "Fraud"]); plt.yticks([0, 1], ["Legit", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix (thr={thr})")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150); plt.close()

    # ---- Random Forest feature importance -----------------------------------
    rf_rows = res[res["model"] == "Random Forest"].sort_values("val_pr_auc", ascending=False)
    rf_pipe = runs[("Random Forest", rf_rows.iloc[0]["scenario"])][0]
    names = SCALE_COLS + [c for c in features if c not in SCALE_COLS]  # ColumnTransformer order
    imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False)
    imp.to_csv(f"{OUT_DIR}/feature_importance.csv", header=["importance"])
    plt.figure(figsize=(6, 5))
    imp.head(15)[::-1].plot(kind="barh")
    plt.title("Top 15 features (Random Forest)"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150); plt.close()

    # ---- save model + demo samples ------------------------------------------
    joblib.dump({"pipeline": pipe, "threshold": thr, "features": features,
                 "model_name": key[0], "scenario": key[1]}, f"{MODEL_DIR}/best_model.joblib")
    # Demo samples: the 3 fraud cases the model is most sure about + 3 legitimate cases
    test_df = X_test.copy(); test_df[TARGET] = y_test.values; test_df["_p"] = p_test
    samples = []
    for label, k in [(1, 3), (0, 3)]:
        part = test_df[test_df[TARGET] == label]
        part = part.sort_values("_p", ascending=False).head(k) if label else part.head(k)
        for _, r in part.iterrows():
            samples.append({"label": "Fraud" if label else "Legitimate",
                            "values": [float(r[c]) for c in features]})
    with open(f"{MODEL_DIR}/sample_transactions.json", "w") as f:
        json.dump(samples, f)
    with open(f"{OUT_DIR}/final_report.json", "w") as f:
        json.dump({"model": key[0], "scenario": key[1], "threshold": thr, **final}, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}/ and tables/plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
