# 🚨 Early Warning System — Risk Management Portfolio

Sistem deteksi dini risiko terintegrasi yang mencakup tiga domain: **Credit Risk**, **Market Risk**, dan **Operational Risk**. Dibangun dengan Python, Scikit-learn, dan divisualisasikan menggunakan Power BI / Tableau.

---

## 🏗️ Arsitektur Sistem

```
Data Sources → ETL Pipeline → Feature Engineering → ML Models → Risk Scoring → Dashboard Alert
```

---

## 📁 Struktur Folder

```
early-warning-system/
├── data/
│   ├── raw/
│   │   ├── credit/          ← German Credit Dataset (UCI)
│   │   ├── market/          ← Yahoo Finance historical prices
│   │   └── operational/     ← Transaction / Fraud data
│   └── processed/           ← Data setelah feature engineering
├── src/
│   ├── data_ingestion.py    ← ETL pipeline (Fase 1)
│   ├── credit_risk/         ← Model & feature untuk credit risk
│   ├── market_risk/         ← Model & feature untuk market risk
│   └── operational_risk/    ← Model & feature untuk fraud detection
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelling.ipynb
├── dashboard/               ← File Power BI (.pbix)
├── reports/                 ← Laporan dan dokumentasi
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Jalankan Data Ingestion (Fase 1)
```bash
python src/data_ingestion.py
```

### 3. Catatan Dataset

| Domain | Dataset | Cara Mendapatkan |
|---|---|---|
| Credit Risk | German Credit | Auto-download via script |
| Market Risk | IHSG + LQ45 | Auto-download via `yfinance` |
| Operational Risk | Synthetic Transaction | Auto-generate via script |

> Untuk menggunakan dataset IEEE-CIS Fraud yang asli: download dari [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection), lalu simpan `train_transaction.csv` ke `data/raw/operational/ieee_fraud_train.csv`.

---

## 📊 Domain Risiko

### 1. Credit Risk
- **Tujuan:** Prediksi kemungkinan gagal bayar debitur
- **Dataset:** German Credit Dataset (1.000 nasabah, 20 fitur)
- **Model:** Logistic Regression, XGBoost
- **Output:** Risk Score 0–100, flag WARNING/CRITICAL

### 2. Market Risk
- **Tujuan:** Deteksi volatilitas ekstrem dan potensi drawdown
- **Dataset:** Harga historis IHSG + saham LQ45 (5 tahun)
- **Model:** Rolling Volatility, Isolation Forest, GARCH
- **Output:** Volatility Alert, VaR breach flag

### 3. Operational Risk
- **Tujuan:** Deteksi transaksi anomali / fraud
- **Dataset:** 50.000 transaksi synthetic (2,5% fraud)
- **Model:** Isolation Forest, Autoencoder, DBSCAN
- **Output:** Fraud Probability Score, anomaly flag

---

## 📈 Tech Stack

- **Language:** Python 3.10+
- **ML Library:** Scikit-learn, XGBoost
- **Data:** Pandas, NumPy, yfinance, SQLAlchemy
- **Visualization:** Power BI / Tableau, Matplotlib, Seaborn
- **Version Control:** Git + GitHub

---

## 👤 Author

Portofolio proyek untuk posisi **Risk Data Scientist / Credit Risk Analyst**

---

*Proyek ini dibuat untuk tujuan edukasi dan portofolio profesional.*