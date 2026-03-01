"""
==============================================================
 Early Warning System — Feature Engineering
 Domain: CREDIT RISK
 Input : data/raw/credit/german_credit_raw.csv
 Output: data/processed/credit/credit_features.csv
==============================================================

Fitur yang dibangun:
  1. Encoding kategorikal (Ordinal + One-Hot)
  2. Financial Ratios (Debt Burden, Savings Coverage)
  3. Risk Flags (high amount, long duration, dll)
  4. Composite Risk Score (rule-based, sebelum ML)
==============================================================
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Path ──────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH    = os.path.join(BASE_DIR, "data", "raw", "credit", "german_credit_raw.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "processed", "credit")
OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "credit_features.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────
# STEP 1: LOAD & INSPECT
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log(f"✅ Loaded {len(df):,} rows, {df.shape[1]} columns")
    log(f"   Target distribution:\n{df['target'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────
# STEP 2: DECODE CATEGORICAL FEATURES
# Kode A11, A32, dll → nilai yang bermakna
# ─────────────────────────────────────────
def decode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Checking account status (saldo rekening giro)
    checking_map = {
        "A11": "negative",    # < 0 DM
        "A12": "low",         # 0–200 DM
        "A13": "moderate",    # >= 200 DM
        "A14": "no_account",  # tidak punya rekening
    }
    checking_risk = {"negative": 3, "low": 2, "moderate": 1, "no_account": 2}

    # Credit history
    history_map = {
        "A30": "no_credits",       # tidak ada kredit / semua dibayar tepat waktu
        "A31": "all_paid",
        "A32": "existing_paid",    # kredit berjalan / dibayar tepat waktu
        "A33": "delayed",          # pernah terlambat
        "A34": "critical",         # akun kritis / kredit lain
    }
    history_risk = {"no_credits": 1, "all_paid": 0, "existing_paid": 1, "delayed": 2, "critical": 3}

    # Savings account
    savings_map = {
        "A61": "low",          # < 100 DM
        "A62": "moderate",     # 100–500 DM
        "A63": "good",         # 500–1000 DM
        "A64": "high",         # >= 1000 DM
        "A65": "unknown",
    }
    savings_risk = {"low": 3, "moderate": 2, "good": 1, "high": 0, "unknown": 2}

    # Employment duration
    employment_map = {
        "A71": "unemployed",
        "A72": "less_1yr",
        "A73": "1to4yr",
        "A74": "4to7yr",
        "A75": "over7yr",
    }
    employment_stability = {"unemployed": 0, "less_1yr": 1, "1to4yr": 2, "4to7yr": 3, "over7yr": 4}

    # Apply mappings
    df["checking_status"]     = df["checking_account"].map(checking_map).fillna("unknown")
    df["checking_risk_score"] = df["checking_account"].map(checking_risk).fillna(2)

    df["credit_history_label"] = df["credit_history"].map(history_map).fillna("unknown")
    df["history_risk_score"]   = df["credit_history"].map(history_risk).fillna(2)

    df["savings_label"]      = df["savings_account"].map(savings_map).fillna("unknown")
    df["savings_risk_score"] = df["savings_account"].map(savings_risk).fillna(2)

    df["employment_label"]       = df["employment_since"].map(employment_map).fillna("unknown")
    df["employment_stability"]   = df["employment_since"].map(employment_stability).fillna(1)

    log("✅ Categorical features decoded")
    return df


# ─────────────────────────────────────────
# STEP 3: FINANCIAL RATIO FEATURES
# ─────────────────────────────────────────
def build_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Debt Burden Index: estimasi beban cicilan per bulan relatif terhadap kredit
    # Semakin tinggi installment_rate dan duration, semakin berat
    df["monthly_installment_est"] = (
        df["credit_amount"] * (df["installment_rate"] / 100)
    )

    # Debt-to-Duration Ratio: kredit besar dengan tenor pendek = beban berat
    df["debt_duration_ratio"] = df["credit_amount"] / df["duration"].replace(0, 1)

    # Age-adjusted risk: nasabah muda dengan kredit besar = lebih berisiko
    df["credit_per_age"] = df["credit_amount"] / df["age"].replace(0, 1)

    # Kredit per tanggungan
    df["credit_per_dependent"] = df["credit_amount"] / df["num_dependents"].replace(0, 1)

    # Existing credit load
    df["credit_load_index"] = df["existing_credits"] * df["installment_rate"]

    log("✅ Financial ratio features built")
    return df


# ─────────────────────────────────────────
# STEP 4: BINARY RISK FLAGS
# Domain knowledge: kondisi yang secara historis meningkatkan risiko
# ─────────────────────────────────────────
def build_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    credit_median = df["credit_amount"].median()

    df["flag_high_credit"]        = (df["credit_amount"] > credit_median * 1.5).astype(int)
    df["flag_long_duration"]      = (df["duration"] > 36).astype(int)          # lebih dari 3 tahun
    df["flag_young_borrower"]     = (df["age"] < 25).astype(int)
    df["flag_negative_checking"]  = (df["checking_status"] == "negative").astype(int)
    df["flag_low_savings"]        = (df["savings_risk_score"] >= 3).astype(int)
    df["flag_unstable_employment"]= (df["employment_stability"] <= 1).astype(int)
    df["flag_critical_history"]   = (df["history_risk_score"] >= 3).astype(int)
    df["flag_high_installment"]   = (df["installment_rate"] >= 4).astype(int)

    # Multi-flag: nasabah dengan 4+ flag = very high risk
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["total_risk_flags"] = df[flag_cols].sum(axis=1)
    df["flag_very_high_risk"] = (df["total_risk_flags"] >= 4).astype(int)

    log(f"✅ Risk flags built ({len(flag_cols)} flags)")
    return df


# ─────────────────────────────────────────
# STEP 5: COMPOSITE RISK SCORE (Rule-Based)
# Sebelum ML — sebagai baseline dan interpretasi
# ─────────────────────────────────────────
def build_composite_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalisasi fitur numerik ke 0–1 untuk scoring
    def minmax(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    score = (
        minmax(df["checking_risk_score"])   * 0.20 +
        minmax(df["history_risk_score"])    * 0.20 +
        minmax(df["savings_risk_score"])    * 0.15 +
        minmax(df["debt_duration_ratio"])   * 0.15 +
        minmax(df["total_risk_flags"])      * 0.15 +
        (1 - minmax(df["employment_stability"])) * 0.10 +
        minmax(df["credit_per_age"])        * 0.05
    )

    # Skalakan ke 0–100
    df["rule_based_risk_score"] = (score * 100).round(2)

    # Traffic Light Label
    df["risk_label"] = pd.cut(
        df["rule_based_risk_score"],
        bins=[0, 33, 66, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True
    )

    log("✅ Composite risk score built")
    log(f"   Distribution:\n{df['risk_label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────
# STEP 6: SELECT FINAL FEATURES
# ─────────────────────────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:

    # Fitur numerik yang akan dipakai model ML di Fase 3
    ml_features = [
        # Original numerik
        "duration", "credit_amount", "installment_rate",
        "age", "existing_credits", "num_dependents",
        # Encoded risk scores
        "checking_risk_score", "history_risk_score",
        "savings_risk_score", "employment_stability",
        # Engineered ratios
        "monthly_installment_est", "debt_duration_ratio",
        "credit_per_age", "credit_per_dependent", "credit_load_index",
        # Risk flags
        "flag_high_credit", "flag_long_duration", "flag_young_borrower",
        "flag_negative_checking", "flag_low_savings",
        "flag_unstable_employment", "flag_critical_history",
        "flag_high_installment", "total_risk_flags",
        # Composite score
        "rule_based_risk_score",
        # Target
        "target", "risk_label"
    ]

    # Ambil hanya kolom yang ada
    available = [c for c in ml_features if c in df.columns]
    return df[available]


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_credit_feature_engineering():
    print("\n" + "="*60)
    print("  CREDIT RISK — Feature Engineering")
    print("="*60)

    df = load_data(INPUT_PATH)
    df = decode_categoricals(df)
    df = build_financial_ratios(df)
    df = build_risk_flags(df)
    df = build_composite_risk_score(df)
    df = select_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    log(f"\n💾 Output saved → {OUTPUT_PATH}")
    log(f"   Shape: {df.shape}")
    log(f"   Features: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = run_credit_feature_engineering()
    print("\n📊 Sample output:")
    print(df[["credit_amount", "duration", "rule_based_risk_score", "risk_label", "target"]].head(10).to_string(index=False))