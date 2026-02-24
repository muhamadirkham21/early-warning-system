"""
==============================================================
 Early Warning System - Data Ingestion Pipeline
 Fase 1: Download & simpan raw data untuk 3 domain risiko
==============================================================
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from urllib.request import urlretrieve
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

CREDIT_DIR      = os.path.join(RAW_DIR, "credit")
MARKET_DIR      = os.path.join(RAW_DIR, "market")
OPERATIONAL_DIR = os.path.join(RAW_DIR, "operational")

# Tickers untuk Market Risk (IHSG + saham LQ45 pilihan)
MARKET_TICKERS = [
    "^JKSE",    # IHSG
    "BBCA.JK",  # BCA
    "BBRI.JK",  # BRI
    "TLKM.JK",  # Telkom
    "ASII.JK",  # Astra
    "BMRI.JK",  # Mandiri
]

MARKET_START = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")
MARKET_END   = datetime.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────
def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────
# 1. CREDIT RISK — German Credit Dataset
# ─────────────────────────────────────────
def ingest_credit_data():
    """
    Download German Credit Dataset dari UCI ML Repository.
    Dataset berisi 1000 baris, 20 fitur, target: 1=Good, 2=Bad credit.
    """
    log("📥 [Credit] Downloading German Credit Dataset...")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Column names sesuai dokumentasi UCI
    columns = [
        "checking_account", "duration", "credit_history", "purpose",
        "credit_amount", "savings_account", "employment_since",
        "installment_rate", "personal_status", "other_debtors",
        "residence_since", "property", "age", "other_installment_plans",
        "housing", "existing_credits", "job", "num_dependents",
        "telephone", "foreign_worker", "target"
    ]
    
    save_path = os.path.join(CREDIT_DIR, "german_credit_raw.csv")
    
    try:
        df = pd.read_csv(url, sep=" ", header=None, names=columns)
        
        # target: 1=Good → 0, 2=Bad → 1 (konversi ke binary label)
        df["target"] = df["target"].map({1: 0, 2: 1})
        
        df.to_csv(save_path, index=False)
        log(f"✅ [Credit] Saved {len(df):,} rows → {save_path}")
        return df
    
    except Exception as e:
        log(f"❌ [Credit] Error: {e}")
        log("   Membuat synthetic credit data sebagai fallback...")
        return _generate_synthetic_credit()


def _generate_synthetic_credit() -> pd.DataFrame:
    """Fallback: buat synthetic data jika UCI tidak bisa diakses."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "checking_account":      np.random.choice(["A11","A12","A13","A14"], n),
        "duration":              np.random.randint(6, 72, n),
        "credit_history":        np.random.choice(["A30","A31","A32","A33","A34"], n),
        "purpose":               np.random.choice(["A40","A41","A42","A43"], n),
        "credit_amount":         np.random.randint(500, 20000, n),
        "savings_account":       np.random.choice(["A61","A62","A63","A64","A65"], n),
        "employment_since":      np.random.choice(["A71","A72","A73","A74","A75"], n),
        "installment_rate":      np.random.randint(1, 5, n),
        "personal_status":       np.random.choice(["A91","A92","A93","A94"], n),
        "age":                   np.random.randint(18, 75, n),
        "existing_credits":      np.random.randint(1, 5, n),
        "num_dependents":        np.random.randint(1, 3, n),
        "target":                np.random.choice([0, 1], n, p=[0.70, 0.30]),
    })
    save_path = os.path.join(CREDIT_DIR, "german_credit_raw.csv")
    df.to_csv(save_path, index=False)
    log(f"✅ [Credit] Synthetic data saved → {save_path}")
    return df


# ─────────────────────────────────────────
# 2. MARKET RISK — Yahoo Finance (yfinance)
# ─────────────────────────────────────────
def ingest_market_data():
    """
    Download data harga saham historis via yfinance.
    Simpan per ticker dan gabungkan dalam satu file ringkasan.
    """
    log("📥 [Market] Downloading historical price data...")

    all_close = {}

    for ticker in MARKET_TICKERS:
        try:
            df = yf.download(ticker, start=MARKET_START, end=MARKET_END, progress=False)
            
            if df.empty:
                log(f"   ⚠️  {ticker}: no data returned, skipping.")
                continue
            
            df.reset_index(inplace=True)
            
            # Flatten MultiIndex columns jika ada
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(col).strip("_") for col in df.columns]
            
            save_path = os.path.join(MARKET_DIR, f"{ticker.replace('^','IDX_').replace('.JK','')}.csv")
            df.to_csv(save_path, index=False)
            
            # Ambil kolom Close untuk ringkasan
            close_col = [c for c in df.columns if "Close" in c or "close" in c]
            if close_col:
                all_close[ticker] = df.set_index(df.columns[0])[close_col[0]]
            
            log(f"   ✅ {ticker}: {len(df):,} rows saved")

        except Exception as e:
            log(f"   ❌ {ticker}: {e}")

    # Simpan combined closing prices
    if all_close:
        combined = pd.DataFrame(all_close)
        combined.index.name = "Date"
        combined.to_csv(os.path.join(MARKET_DIR, "combined_close_prices.csv"))
        log(f"✅ [Market] Combined close prices saved ({combined.shape})")
        return combined
    
    return pd.DataFrame()


# ─────────────────────────────────────────
# 3. OPERATIONAL RISK — Synthetic Fraud Data
# ─────────────────────────────────────────
def ingest_operational_data():
    """
    Generate synthetic transaction data yang menyerupai pola fraud dataset.
    Catatan: IEEE-CIS Fraud dataset asli memerlukan login Kaggle.
    Dataset ini bisa diganti dengan data asli jika sudah didownload manual.
    
    Struktur: TransactionID, TransactionDT, TransactionAmt, 
              ProductCD, card_type, is_fraud
    """
    log("📥 [Operational] Generating synthetic transaction data...")

    kaggle_path = os.path.join(OPERATIONAL_DIR, "ieee_fraud_train.csv")
    if os.path.exists(kaggle_path):
        log(f"   📂 File Kaggle ditemukan, menggunakan data asli.")
        df = pd.read_csv(kaggle_path)
        log(f"✅ [Operational] Loaded {len(df):,} rows dari Kaggle dataset")
        return df

    # ── Synthetic generation ──
    np.random.seed(42)
    n = 50_000

    transaction_dt = np.sort(np.random.randint(86400, 86400 * 365, n))  # detik dalam setahun

    # Pola normal: sebagian besar transaksi kecil di jam kerja
    amounts_normal = np.abs(np.random.lognormal(mean=4.0, sigma=1.2, size=n))

    # Inject fraud (2.5% dari data)
    fraud_mask = np.random.choice([0, 1], n, p=[0.975, 0.025]).astype(bool)
    amounts_normal[fraud_mask] *= np.random.uniform(3, 15, fraud_mask.sum())  # fraud = amount jauh lebih besar

    df = pd.DataFrame({
        "TransactionID":  range(1, n + 1),
        "TransactionDT":  transaction_dt,
        "TransactionAmt": np.round(amounts_normal, 2),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        "card_type":      np.random.choice(["visa", "mastercard", "discover", "amex"], n, p=[0.5, 0.3, 0.1, 0.1]),
        "card_bank":      np.random.choice(["BCA", "BRI", "Mandiri", "BNI", "CIMB"], n),
        "hour_of_day":    (transaction_dt % 86400) // 3600,
        "is_fraud":       fraud_mask.astype(int),
    })

    save_path = os.path.join(OPERATIONAL_DIR, "synthetic_transactions.csv")
    df.to_csv(save_path, index=False)
    fraud_count = df["is_fraud"].sum()
    log(f"✅ [Operational] {len(df):,} transaksi ({fraud_count:,} fraud / {fraud_count/n:.1%}) → {save_path}")
    return df


# ─────────────────────────────────────────
# 4. GENERATE INGESTION REPORT
# ─────────────────────────────────────────
def generate_report(credit_df, market_df, operational_df):
    report = f"""
=================================================================
  EARLY WARNING SYSTEM — Data Ingestion Report
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=================================================================

[1] CREDIT RISK
  Dataset   : German Credit (UCI ML Repository)
  Rows      : {len(credit_df):,}
  Columns   : {credit_df.shape[1]}
  Fraud Rate: {credit_df['target'].mean():.1%} (Bad Credit)

[2] MARKET RISK  
  Tickers   : {', '.join(MARKET_TICKERS)}
  Period    : {MARKET_START} → {MARKET_END}
  Shape     : {market_df.shape if not market_df.empty else 'N/A'}

[3] OPERATIONAL RISK
  Dataset   : Synthetic Transaction Data
  Rows      : {len(operational_df):,}
  Fraud Rate: {operational_df['is_fraud'].mean():.1%}

STATUS: ✅ Ingestion Complete
=================================================================
"""
    print(report)
    
    report_path = os.path.join(BASE_DIR, "reports", "ingestion_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    log(f"📄 Report saved → {report_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EWS Data Ingestion Pipeline — Starting...")
    print("=" * 60)

    # Pastikan folder ada
    for folder in [CREDIT_DIR, MARKET_DIR, OPERATIONAL_DIR]:
        os.makedirs(folder, exist_ok=True)

    # Jalankan ingestion
    credit_df      = ingest_credit_data()
    market_df      = ingest_market_data()
    operational_df = ingest_operational_data()

    # Buat report
    generate_report(credit_df, market_df, operational_df)

    print("\n🎉 Fase 1 selesai! Semua data tersimpan di folder data/raw/")
    print("   Selanjutnya: jalankan notebook 01_EDA.ipynb\n")