"""
==============================================================
 Early Warning System — Modelling
 Domain : CREDIT RISK
 Model  : Logistic Regression (baseline) + XGBoost (utama)
 Input  : data/processed/credit/credit_features.csv
 Output : models/credit/  +  data/processed/credit/credit_scores.csv
==============================================================
Metrik utama: AUC-ROC, Recall (kita prioritaskan tangkap bad credit)
==============================================================
"""

import os, sys, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "credit", "credit_features.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "credit")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "credit")
REPORT_DIR  = os.path.join(BASE_DIR, "reports", "modelling")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def log(msg): print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ── Feature columns untuk ML ───────────────────────────────
FEATURE_COLS = [
    "duration", "credit_amount", "installment_rate",
    "age", "existing_credits", "num_dependents",
    "checking_risk_score", "history_risk_score",
    "savings_risk_score", "employment_stability",
    "monthly_installment_est", "debt_duration_ratio",
    "credit_per_age", "credit_per_dependent", "credit_load_index",
    "flag_high_credit", "flag_long_duration", "flag_young_borrower",
    "flag_negative_checking", "flag_low_savings",
    "flag_unstable_employment", "flag_critical_history",
    "flag_high_installment", "total_risk_flags",
]
TARGET = "target"


# ─────────────────────────────────────────
# STEP 1: LOAD & SPLIT
# ─────────────────────────────────────────
def load_and_split(path):
    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"Train: {X_train.shape} | Test: {X_test.shape}")
    log(f"Bad credit rate — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test, df, available


# ─────────────────────────────────────────
# STEP 2: BASELINE — Logistic Regression
# ─────────────────────────────────────────
def train_baseline(X_train, y_train, X_test, y_test):
    log("Training Logistic Regression (baseline)...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, prob)
    log(f"Baseline AUC-ROC: {auc:.4f}")
    return pipe, auc


# ─────────────────────────────────────────
# STEP 3: MAIN MODEL — XGBoost
# ─────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    log("Training XGBoost...")

    # scale_pos_weight untuk handle imbalance
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = spw,
        eval_metric       = "auc",
        random_state      = 42,
        verbosity         = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    prob = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, prob)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    log(f"XGBoost AUC-ROC : {auc:.4f}")
    log(f"CV AUC (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, prob, auc


# ─────────────────────────────────────────
# STEP 4: OPTIMAL THRESHOLD
# Kita cari threshold yang maximise Recall
# karena lebih baik false alarm drpd miss
# ─────────────────────────────────────────
def find_optimal_threshold(y_test, prob, min_recall=0.75):
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    # Threshold dengan recall >= min_recall dan precision tertinggi
    mask = recall[:-1] >= min_recall
    if mask.any():
        best_idx = np.argmax(precision[:-1][mask])
        candidates = thresholds[mask]
        threshold = candidates[best_idx]
    else:
        threshold = 0.40  # fallback

    y_pred = (prob >= threshold).astype(int)
    log(f"Optimal threshold: {threshold:.3f}")
    log(f"\n{classification_report(y_test, y_pred, target_names=['Good','Bad'])}")
    return threshold


# ─────────────────────────────────────────
# STEP 5: GENERATE RISK SCORES
# ─────────────────────────────────────────
def generate_credit_scores(model, df, feature_cols, threshold):
    X_all = df[feature_cols]
    prob  = model.predict_proba(X_all)[:, 1]

    result = df.copy()
    result["ml_default_prob"]    = prob.round(4)
    result["ml_risk_score"]      = (prob * 100).round(1)
    result["ml_risk_label"]      = pd.cut(
        result["ml_risk_score"],
        bins=[0, 33, 66, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True
    )
    result["ews_signal"] = "NORMAL"
    result.loc[result["ml_default_prob"] >= threshold,        "ews_signal"] = "WARNING"
    result.loc[result["ml_default_prob"] >= threshold + 0.20, "ews_signal"] = "CRITICAL"

    log(f"\n  EWS Signal Distribution:")
    log(f"\n{result['ews_signal'].value_counts().to_string()}")
    return result


# ─────────────────────────────────────────
# STEP 6: VISUALISASI
# ─────────────────────────────────────────
def plot_results(model, X_test, y_test, prob_base, prob_xgb, feature_cols, threshold):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Credit Risk Model — Evaluation Dashboard", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # [1] ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    for label, prob, color in [
        ("XGBoost",          prob_xgb,  "#e74c3c"),
        ("Logistic Baseline",prob_base, "#3498db"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} (AUC={auc:.3f})")
    ax1.plot([0,1],[0,1],"--", color="gray", linewidth=1)
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=8)

    # [2] Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    p, r, t = precision_recall_curve(y_test, prob_xgb)
    ax2.plot(r, p, color="#e74c3c", linewidth=2)
    ax2.axvline(0.75, color="navy", linestyle="--", linewidth=1.5, label="Min Recall = 75%")
    ax2.set_title("Precision-Recall Curve (XGBoost)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8)

    # [3] Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    y_pred = (prob_xgb >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax3,
                xticklabels=["Good","Bad"], yticklabels=["Good","Bad"],
                cbar=False, linewidths=0.5)
    ax3.set_title(f"Confusion Matrix\n(threshold={threshold:.2f})")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    # [4] Feature Importance
    ax4 = fig.add_subplot(gs[1, :2])
    fi = pd.Series(model.feature_importances_, index=feature_cols)
    fi = fi.sort_values(ascending=True).tail(15)
    colors = ["#e74c3c" if v > fi.quantile(0.8) else "#3498db" for v in fi.values]
    ax4.barh(fi.index, fi.values, color=colors, edgecolor="white")
    ax4.set_title("Top 15 Feature Importance (XGBoost)")
    ax4.set_xlabel("Importance Score")

    # [5] Risk Score Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    scores_good = prob_xgb[y_test == 0] * 100
    scores_bad  = prob_xgb[y_test == 1] * 100
    ax5.hist(scores_good, bins=30, alpha=0.6, color="#2ecc71",  label="Good Credit")
    ax5.hist(scores_bad,  bins=30, alpha=0.7, color="#e74c3c", label="Bad Credit")
    ax5.axvline(threshold*100, color="navy", linestyle="--",
                linewidth=2, label=f"Threshold ({threshold*100:.0f})")
    ax5.set_title("Risk Score Distribution")
    ax5.set_xlabel("Risk Score (0–100)")
    ax5.legend(fontsize=8)

    plt.savefig(os.path.join(REPORT_DIR, "credit_model_evaluation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("Chart saved → reports/modelling/credit_model_evaluation.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_credit_modelling():
    print("\n" + "="*60)
    print("  CREDIT RISK — Modelling & Early Warning")
    print("="*60)

    X_train, X_test, y_train, y_test, df, feature_cols = load_and_split(INPUT_PATH)

    baseline, prob_base = train_baseline(X_train, y_train, X_test, y_test)[:2]
    prob_base = baseline.predict_proba(X_test)[:, 1]

    xgb_model, prob_xgb, auc = train_xgboost(X_train, y_train, X_test, y_test)
    threshold = find_optimal_threshold(y_test, prob_xgb)

    plot_results(xgb_model, X_test, y_test, prob_base, prob_xgb, feature_cols, threshold)

    # Score seluruh dataset
    result = generate_credit_scores(xgb_model, df, feature_cols, threshold)
    result.to_csv(os.path.join(OUTPUT_DIR, "credit_scores.csv"), index=False)

    # Simpan model & threshold
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_credit.pkl"))
    joblib.dump({"threshold": threshold, "features": feature_cols,
                 "auc": auc, "trained_at": str(datetime.now())},
                os.path.join(MODEL_DIR, "credit_metadata.pkl"))
    log("Model saved → models/credit/xgb_credit.pkl")

    return xgb_model, threshold, auc


if __name__ == "__main__":
    run_credit_modelling()