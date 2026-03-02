"""
==============================================================
 Early Warning System — Modelling
 Domain : OPERATIONAL RISK (Fraud Detection)
 Model  : XGBoost + SMOTE (handle class imbalance)
 Input  : data/processed/operational/operational_features.csv
 Output : models/operational/  +  data/processed/operational/operational_scores.csv
==============================================================
Tantangan utama: class imbalance ~1:40
Strategi: SMOTE untuk oversample minority class di training set
Metrik: AUC-ROC, F1, Recall (bukan Accuracy)
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Coba import SMOTE, fallback ke class_weight jika tidak ada
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  [INFO] imbalanced-learn tidak tersedia. Menggunakan scale_pos_weight sebagai alternatif.")
    print("         Install dengan: pip install imbalanced-learn")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "operational", "operational_features.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "operational")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "operational")
REPORT_DIR  = os.path.join(BASE_DIR, "reports", "modelling")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def log(msg): print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "week_of_year",
    "flag_odd_hour", "flag_weekend", "flag_late_night",
    "log_amount", "amount_zscore", "amount_zscore_product",
    "amount_zscore_bank", "amount_percentile",
    "flag_high_amount_95", "flag_high_amount_99", "flag_extreme_zscore",
    "card_mean_amt", "card_std_amt", "amt_deviation_from_card_mean",
    "bank_mean_amt", "bank_txn_count",
    "hourly_volume_ratio", "product_fraud_rate",
    "flag_night_high_amount", "flag_extreme_deviation",
    "flag_anomalous_product", "total_anomaly_flags",
]
CATEGORICAL_COLS = ["card_type", "card_bank", "ProductCD"]
TARGET = "is_fraud"


# ─────────────────────────────────────────
# STEP 1: LOAD & ENCODE
# ─────────────────────────────────────────
def load_and_encode(path):
    df = pd.read_csv(path)
    log(f"Loaded: {df.shape} | Fraud rate: {df[TARGET].mean():.2%}")

    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    cat_encoded = [c + "_enc" for c in CATEGORICAL_COLS if c + "_enc" in df.columns]
    all_features = [c for c in FEATURE_COLS + cat_encoded if c in df.columns]

    X = df[all_features].fillna(0)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"Train: {X_train.shape} | Fraud in train: {y_train.sum():,} ({y_train.mean():.2%})")
    return X_train, X_test, y_train, y_test, df, all_features, encoders


# ─────────────────────────────────────────
# STEP 2: HANDLE IMBALANCE
# ─────────────────────────────────────────
def handle_imbalance(X_train, y_train):
    if HAS_SMOTE:
        log("Applying SMOTE (oversampling minority class)...")
        smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        log(f"After SMOTE — Total: {len(X_res):,} | Fraud: {y_res.sum():,} ({y_res.mean():.2%})")
    else:
        log("Using scale_pos_weight instead of SMOTE...")
        X_res, y_res = X_train.copy(), y_train.copy()

    return X_res, y_res


# ─────────────────────────────────────────
# STEP 3: TRAIN XGBOOST
# ─────────────────────────────────────────
def train_xgboost(X_train, y_train, X_res, y_res, X_test, y_test):
    log("Training XGBoost + SMOTE...")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw = neg / pos  # fallback weight jika tidak pakai SMOTE

    model = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.7,
        min_child_weight  = 10,
        scale_pos_weight  = 1 if HAS_SMOTE else spw,
        eval_metric       = "aucpr",   # PR-AUC lebih informatif untuk imbalanced data
        random_state      = 42,
        verbosity         = 0,
    )
    model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    prob     = model.predict_proba(X_test)[:, 1]
    auc_roc  = roc_auc_score(y_test, prob)
    auc_pr   = average_precision_score(y_test, prob)

    log(f"AUC-ROC : {auc_roc:.4f}")
    log(f"AUC-PR  : {auc_pr:.4f}  ← metrik utama untuk imbalanced")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    log(f"CV AUC-ROC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, prob, auc_roc, auc_pr


# ─────────────────────────────────────────
# STEP 4: OPTIMAL THRESHOLD
# Prioritaskan Recall tinggi untuk fraud
# ─────────────────────────────────────────
def find_optimal_threshold(y_test, prob, target_recall=0.80):
    precision, recall, thresholds = precision_recall_curve(y_test, prob)
    # Cari threshold dengan recall >= target & precision tertinggi
    mask = recall[:-1] >= target_recall
    if mask.any():
        best_idx  = np.argmax(precision[:-1][mask])
        threshold = thresholds[mask][best_idx]
    else:
        threshold = 0.30  # fallback agresif untuk fraud

    y_pred = (prob >= threshold).astype(int)
    log(f"Optimal threshold: {threshold:.3f}")
    log(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")
    return threshold


# ─────────────────────────────────────────
# STEP 5: GENERATE SCORES
# ─────────────────────────────────────────
def generate_operational_scores(model, df, feature_cols, threshold):
    X_all = df[feature_cols].fillna(0)
    prob  = model.predict_proba(X_all)[:, 1]

    result = df.copy()
    result["ml_fraud_prob"]      = prob.round(4)
    result["ml_risk_score"]      = (prob * 100).round(1)
    result["ews_signal"] = "NORMAL"
    result.loc[result["ml_fraud_prob"] >= threshold,        "ews_signal"] = "WARNING"
    result.loc[result["ml_fraud_prob"] >= threshold + 0.25, "ews_signal"] = "CRITICAL"

    log(f"\n  EWS Signal Distribution:")
    log(f"\n{result['ews_signal'].value_counts().to_string()}")

    # Precision check: dari yang WARNING/CRITICAL, berapa % memang fraud?
    flagged = result[result["ews_signal"] != "NORMAL"]
    if len(flagged) > 0 and "is_fraud" in result.columns:
        precision_flagged = flagged["is_fraud"].mean()
        log(f"\n  Precision on flagged transactions: {precision_flagged:.2%}")

    return result


# ─────────────────────────────────────────
# STEP 6: VISUALISASI
# ─────────────────────────────────────────
def plot_results(model, y_test, prob, feature_cols, threshold):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Operational Risk (Fraud) Model — Evaluation Dashboard",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # [1] PR Curve (lebih relevan dari ROC untuk imbalanced)
    ax1 = fig.add_subplot(gs[0, 0])
    p, r, t = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax1.plot(r, p, color="#e74c3c", linewidth=2, label=f"XGBoost (AP={ap:.3f})")
    ax1.axhline(y_test.mean(), color="gray", linestyle="--",
                linewidth=1.5, label=f"Baseline ({y_test.mean():.2%})")
    ax1.fill_between(r, p, alpha=0.15, color="#e74c3c")
    ax1.set_title("Precision-Recall Curve\n(lebih informatif untuk imbalanced)")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(fontsize=8)

    # [2] ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax2.plot(fpr, tpr, color="#e74c3c", linewidth=2, label=f"AUC = {auc:.3f}")
    ax2.plot([0,1],[0,1], "--", color="gray", linewidth=1)
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=9)

    # [3] Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    y_pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax3,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"],
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

    # [5] Fraud Probability Distribution
    ax5 = fig.add_subplot(gs[1, 2])
    prob_legit = prob[y_test == 0]
    prob_fraud = prob[y_test == 1]
    ax5.hist(prob_legit, bins=50, alpha=0.6, color="#2ecc71", label="Legit", density=True)
    ax5.hist(prob_fraud, bins=50, alpha=0.7, color="#e74c3c", label="Fraud", density=True)
    ax5.axvline(threshold, color="navy", linestyle="--",
                linewidth=2, label=f"Threshold ({threshold:.2f})")
    ax5.set_title("Fraud Probability Distribution")
    ax5.set_xlabel("P(Fraud)")
    ax5.set_ylabel("Density")
    ax5.legend(fontsize=8)

    plt.savefig(os.path.join(REPORT_DIR, "operational_model_evaluation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    log("Chart saved → reports/modelling/operational_model_evaluation.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_operational_modelling():
    print("\n" + "="*60)
    print("  OPERATIONAL RISK — Modelling & Early Warning")
    print("="*60)

    X_train, X_test, y_train, y_test, df, feature_cols, encoders = load_and_encode(INPUT_PATH)
    X_res, y_res = handle_imbalance(X_train, y_train)
    model, prob, auc_roc, auc_pr = train_xgboost(X_train, y_train, X_res, y_res, X_test, y_test)
    threshold = find_optimal_threshold(y_test, prob)

    plot_results(model, y_test, prob, feature_cols, threshold)

    result = generate_operational_scores(model, df, feature_cols, threshold)
    result.to_csv(os.path.join(OUTPUT_DIR, "operational_scores.csv"), index=False)

    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_operational.pkl"))
    joblib.dump({"threshold": threshold, "features": feature_cols,
                 "encoders": encoders, "auc_roc": auc_roc, "auc_pr": auc_pr,
                 "has_smote": HAS_SMOTE, "trained_at": str(datetime.now())},
                os.path.join(MODEL_DIR, "operational_metadata.pkl"))
    log("Model saved → models/operational/xgb_operational.pkl")

    return model, threshold, auc_roc


if __name__ == "__main__":
    run_operational_modelling()