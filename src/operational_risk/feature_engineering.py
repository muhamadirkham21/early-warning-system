"""
==============================================================
 Early Warning System — Feature Engineering
 Domain: OPERATIONAL RISK (Fraud / Anomaly Detection)
 Input : data/raw/operational/synthetic_transactions.csv
 Output: data/processed/operational/operational_features.csv
==============================================================

Fitur yang dibangun:
  1. Time-based features (jam, hari, pola temporal)
  2. Amount-based features (z-score, rasio terhadap rata-rata)
  3. Behavioral features (frekuensi, pola per kartu/bank)
  4. Anomaly Flags (transaksi yang menyimpang dari pola normal)
  5. Operational Risk Score
==============================================================
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Path ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "raw", "operational", "synthetic_transactions.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "operational")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "operational_features.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    log(f"✅ Loaded {len(df):,} transactions")
    log(f"   Fraud rate: {df['is_fraud'].mean():.2%} ({df['is_fraud'].sum():,} fraud)")
    return df


# ─────────────────────────────────────────
# STEP 2: TIME-BASED FEATURES
# Fraud sering terjadi di jam-jam tidak wajar
# ─────────────────────────────────────────
def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Hour of day (sudah ada dari ingestion, tapi kita pastikan)
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = (df["TransactionDT"] % 86400) // 3600

    # Day of week (0=Mon, 6=Sun) dari TransactionDT
    seconds_per_day = 86400
    df["day_of_week"] = (df["TransactionDT"] // seconds_per_day) % 7

    # Waktu dalam setahun (minggu ke berapa)
    df["week_of_year"] = (df["TransactionDT"] // (seconds_per_day * 7)) % 52 + 1

    # Kategorisasi jam
    df["time_of_day"] = pd.cut(
        df["hour_of_day"],
        bins=[-1, 5, 11, 17, 23],
        labels=["midnight", "morning", "afternoon", "evening"]
    )

    # Flag: transaksi di jam tidak wajar (00:00–05:00)
    df["flag_odd_hour"]    = (df["hour_of_day"] < 6).astype(int)
    df["flag_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["flag_late_night"]  = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 4)).astype(int)

    log("✅ Time features built")
    return df


# ─────────────────────────────────────────
# STEP 3: AMOUNT-BASED FEATURES
# Anomali sering terlihat dari jumlah yang sangat berbeda dari normal
# ─────────────────────────────────────────
def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log transform (mengurangi skewness distribusi amount)
    df["log_amount"] = np.log1p(df["TransactionAmt"])

    # Z-score global (standarisasi)
    mean_amt = df["TransactionAmt"].mean()
    std_amt  = df["TransactionAmt"].std()
    df["amount_zscore"] = (df["TransactionAmt"] - mean_amt) / (std_amt + 1e-9)

    # Z-score per ProductCD (berbeda produk, berbeda ekspektasi)
    df["amount_zscore_product"] = df.groupby("ProductCD")["TransactionAmt"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    # Z-score per bank
    df["amount_zscore_bank"] = df.groupby("card_bank")["TransactionAmt"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    # Percentile rank amount
    df["amount_percentile"] = df["TransactionAmt"].rank(pct=True)

    # Flags berdasarkan amount
    p95 = df["TransactionAmt"].quantile(0.95)
    p99 = df["TransactionAmt"].quantile(0.99)
    df["flag_high_amount_95"] = (df["TransactionAmt"] > p95).astype(int)
    df["flag_high_amount_99"] = (df["TransactionAmt"] > p99).astype(int)
    df["flag_extreme_zscore"] = (df["amount_zscore"].abs() > 3).astype(int)   # > 3 sigma

    log(f"✅ Amount features built | p95={p95:.0f}, p99={p99:.0f}")
    return df


# ─────────────────────────────────────────
# STEP 4: BEHAVIORAL / AGGREGATION FEATURES
# Pola perilaku per entitas (kartu, bank, produk)
# ─────────────────────────────────────────
def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Per card_type ──
    card_stats = df.groupby("card_type")["TransactionAmt"].agg(
        card_mean_amt="mean",
        card_std_amt="std",
        card_txn_count="count"
    ).reset_index()
    df = df.merge(card_stats, on="card_type", how="left")

    # Deviasi dari rata-rata kartu
    df["amt_deviation_from_card_mean"] = (
        df["TransactionAmt"] - df["card_mean_amt"]
    ) / (df["card_std_amt"] + 1e-9)

    # ── Per bank ──
    bank_stats = df.groupby("card_bank")["TransactionAmt"].agg(
        bank_mean_amt="mean",
        bank_std_amt="std",
        bank_txn_count="count"
    ).reset_index()
    df = df.merge(bank_stats, on="card_bank", how="left")

    # ── Transaksi per jam (volume anomali) ──
    hourly_volume = df.groupby("hour_of_day").size().reset_index(name="txn_per_hour_avg")
    df = df.merge(hourly_volume, on="hour_of_day", how="left")

    # Rasio volume jam ini vs rata-rata
    avg_hourly = df["txn_per_hour_avg"].mean()
    df["hourly_volume_ratio"] = df["txn_per_hour_avg"] / (avg_hourly + 1e-9)

    # ── Per ProductCD ──
    prod_fraud_rate = df.groupby("ProductCD")["is_fraud"].mean().reset_index()
    prod_fraud_rate.columns = ["ProductCD", "product_fraud_rate"]
    df = df.merge(prod_fraud_rate, on="ProductCD", how="left")

    log("✅ Behavioral features built")
    return df


# ─────────────────────────────────────────
# STEP 5: COMPOSITE ANOMALY FLAGS
# ─────────────────────────────────────────
def build_anomaly_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Kombinasi flag yang paling sering berkorelasi dengan fraud
    df["flag_night_high_amount"]    = (
        (df["flag_late_night"] == 1) & (df["flag_high_amount_95"] == 1)
    ).astype(int)

    df["flag_extreme_deviation"]    = (
        df["amt_deviation_from_card_mean"].abs() > 3
    ).astype(int)

    df["flag_anomalous_product"]    = (
        df["product_fraud_rate"] > df["product_fraud_rate"].quantile(0.75)
    ).astype(int)

    # Total flag count
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["total_anomaly_flags"] = df[flag_cols].sum(axis=1)

    log(f"✅ Anomaly flags built ({len(flag_cols)} flags)")
    return df


# ─────────────────────────────────────────
# STEP 6: OPERATIONAL RISK SCORE
# ─────────────────────────────────────────
def build_operational_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    score = (
        minmax(df["amount_zscore"].abs())               * 0.25 +
        minmax(df["amount_zscore_product"].abs())       * 0.15 +
        minmax(df["amt_deviation_from_card_mean"].abs())* 0.20 +
        minmax(df["total_anomaly_flags"])               * 0.20 +
        df["flag_late_night"]                           * 0.10 +
        df["flag_high_amount_99"]                       * 0.10
    )

    df["operational_risk_score"] = (score * 100).round(2)

    df["operational_risk_label"] = pd.cut(
        df["operational_risk_score"],
        bins=[0, 33, 66, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True
    )

    log("✅ Operational risk score built")
    log(f"   Distribution:\n{df['operational_risk_label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────
# STEP 7: SELECT FINAL FEATURES
# ─────────────────────────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    ml_features = [
        # IDs
        "TransactionID",
        # Time features
        "hour_of_day", "day_of_week", "week_of_year",
        "flag_odd_hour", "flag_weekend", "flag_late_night",
        # Amount features
        "TransactionAmt", "log_amount", "amount_zscore",
        "amount_zscore_product", "amount_zscore_bank",
        "amount_percentile",
        "flag_high_amount_95", "flag_high_amount_99", "flag_extreme_zscore",
        # Behavioral
        "card_mean_amt", "card_std_amt",
        "amt_deviation_from_card_mean",
        "bank_mean_amt", "bank_txn_count",
        "hourly_volume_ratio", "product_fraud_rate",
        # Categorical (encoded sederhana)
        "card_type", "card_bank", "ProductCD",
        # Anomaly flags
        "flag_night_high_amount", "flag_extreme_deviation",
        "flag_anomalous_product", "total_anomaly_flags",
        # Risk score
        "operational_risk_score", "operational_risk_label",
        # Target
        "is_fraud",
    ]
    available = [c for c in ml_features if c in df.columns]
    return df[available]


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_operational_feature_engineering():
    print("\n" + "="*60)
    print("  OPERATIONAL RISK — Feature Engineering")
    print("="*60)

    df = load_data(INPUT_PATH)
    df = build_time_features(df)
    df = build_amount_features(df)
    df = build_behavioral_features(df)
    df = build_anomaly_flags(df)
    df = build_operational_risk_score(df)
    df = select_features(df)

    df.to_csv(OUTPUT_PATH, index=False)
    log(f"\n💾 Output saved → {OUTPUT_PATH}")
    log(f"   Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = run_operational_feature_engineering()
    print("\n📊 Sample output (fraud cases):")
    fraud_sample = df[df["is_fraud"] == 1][
        ["TransactionAmt", "hour_of_day", "amount_zscore",
         "total_anomaly_flags", "operational_risk_score", "operational_risk_label"]
    ].head(10)
    print(fraud_sample.to_string(index=False))