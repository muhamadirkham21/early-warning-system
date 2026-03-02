"""
==============================================================
 Early Warning System — Modelling
 Domain : MARKET RISK
 Model  : Isolation Forest (anomaly detection) +
          Rolling Threshold (VaR breach detection)
 Input  : data/processed/market/market_features.csv
 Output : models/market/  +  data/processed/market/market_scores.csv
==============================================================
Pendekatan: Market risk bersifat time-series & unsupervised
karena tidak ada label "krisis" yang jelas.
Kita kombinasikan rule-based threshold + Isolation Forest.
==============================================================
"""

import os, sys, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "market", "market_features.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "market")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "market")
REPORT_DIR  = os.path.join(BASE_DIR, "reports", "modelling")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def log(msg): print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

FEATURE_COLS = [
    "vol_5d", "vol_10d", "vol_20d", "vol_60d",
    "ret_5d", "ret_20d",
    "vol_zscore",
    "drawdown", "max_drawdown_60d",
    "var_95_1d", "var_99_1d", "cvar_95_1d",
    "rsi_14", "bb_width",
]


# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[available].dropna()
    log(f"Loaded: {df_clean.shape} | {df_clean.index[0].date()} → {df_clean.index[-1].date()}")
    return df, df_clean, available


# ─────────────────────────────────────────
# STEP 2: ISOLATION FOREST
# Deteksi hari-hari anomali di pasar
# ─────────────────────────────────────────
def train_isolation_forest(df_clean, feature_cols):
    log("Training Isolation Forest...")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  IsolationForest(
            n_estimators  = 200,
            contamination = 0.05,   # ekspektasi ~5% hari anomali
            max_features  = 0.8,
            random_state  = 42,
        ))
    ])
    pipe.fit(df_clean[feature_cols])

    # anomaly score: lebih negatif = lebih anomali
    raw_scores  = pipe.score_samples(df_clean[feature_cols])
    predictions = pipe.predict(df_clean[feature_cols])  # -1=anomali, 1=normal

    # Normalize ke 0–100 (100=paling anomali)
    norm_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    norm_scores = (norm_scores * 100).round(2)

    log(f"Anomaly days detected: {(predictions == -1).sum()} / {len(predictions)} "
        f"({(predictions == -1).mean():.1%})")

    return pipe, norm_scores, predictions


# ─────────────────────────────────────────
# STEP 3: RULE-BASED VaR BREACH DETECTION
# ─────────────────────────────────────────
def detect_var_breaches(df):
    signals = pd.DataFrame(index=df.index)

    # Flags dari feature engineering
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        signals["n_flags"]        = df[flag_cols].sum(axis=1)
    else:
        signals["n_flags"]        = 0

    signals["var_breach_95"]  = df.get("var_breach_95",  pd.Series(0, index=df.index))
    signals["var_breach_99"]  = df.get("var_breach_99",  pd.Series(0, index=df.index))
    signals["is_high_vol"]    = df.get("is_high_vol",    pd.Series(0, index=df.index))
    signals["trend_bearish"]  = df.get("trend_bearish",  pd.Series(0, index=df.index))

    # Consecutive VaR breaches (lebih berbahaya)
    if "var_breach_95" in df.columns:
        signals["consec_var_breach"] = (
            df["var_breach_95"]
            .groupby((df["var_breach_95"] != df["var_breach_95"].shift()).cumsum())
            .cumsum()
        )
    else:
        signals["consec_var_breach"] = 0

    log(f"VaR 95% breach days: {signals['var_breach_95'].sum():.0f}")
    log(f"VaR 99% breach days: {signals['var_breach_99'].sum():.0f}")
    return signals


# ─────────────────────────────────────────
# STEP 4: COMBINED MARKET RISK SCORE & EWS
# ─────────────────────────────────────────
def generate_market_scores(df, df_clean, norm_scores, predictions, signals):
    result = df.copy()

    # Isolation Forest scores (hanya baris yang digunakan model)
    result["if_anomaly_score"] = np.nan
    result["if_is_anomaly"]    = 0
    result.loc[df_clean.index, "if_anomaly_score"] = norm_scores
    result.loc[df_clean.index, "if_is_anomaly"]    = (predictions == -1).astype(int)

    # Gabungkan dengan signals
    for col in signals.columns:
        result[col] = signals[col]

    # Final risk score: weighted combination
    def safe_col(col):
        return result[col].fillna(0) if col in result.columns else pd.Series(0, index=result.index)

    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    result["market_ews_score"] = (
        minmax(safe_col("if_anomaly_score"))  * 0.40 +
        minmax(safe_col("n_flags"))           * 0.20 +
        safe_col("var_breach_99")             * 0.15 +
        safe_col("is_high_vol")               * 0.15 +
        safe_col("trend_bearish")             * 0.10
    ) * 100

    # EWS Signal
    result["ews_signal"] = "NORMAL"
    result.loc[result["market_ews_score"] >= 50, "ews_signal"] = "WARNING"
    result.loc[result["market_ews_score"] >= 75, "ews_signal"] = "CRITICAL"

    log(f"\n  EWS Signal Distribution:")
    log(f"\n{result['ews_signal'].value_counts().to_string()}")
    return result


# ─────────────────────────────────────────
# STEP 5: VISUALISASI
# ─────────────────────────────────────────
def plot_results(result, df_clean, feature_cols):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Market Risk Model — EWS Dashboard", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    colors_map = {"NORMAL": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}

    # [1] EWS Score timeline
    ax1 = fig.add_subplot(gs[0, :])
    score = result["market_ews_score"].dropna()
    signal = result["ews_signal"].reindex(score.index)
    colors = [colors_map.get(s, "gray") for s in signal]

    ax1.fill_between(score.index, score, alpha=0.25, color="#3498db")
    ax1.plot(score.index, score, color="#3498db", linewidth=1, alpha=0.7)
    ax1.scatter(score.index, score, c=colors, s=8, zorder=5, alpha=0.8)
    ax1.axhline(50, color="#f39c12", linestyle="--", linewidth=1.5, label="WARNING (50)")
    ax1.axhline(75, color="#e74c3c", linestyle="--", linewidth=1.5, label="CRITICAL (75)")
    ax1.set_title("Market EWS Score Timeline")
    ax1.set_ylabel("Risk Score (0–100)")
    ax1.legend(fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors_map.items()]
    ax1.legend(handles=legend_elements + [
        plt.Line2D([0],[0], color="#f39c12", linestyle="--", label="WARNING threshold"),
        plt.Line2D([0],[0], color="#e74c3c", linestyle="--", label="CRITICAL threshold"),
    ], fontsize=8, loc="upper left")

    # [2] Anomaly Score Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    anom_scores = result["if_anomaly_score"].dropna()
    ax2.hist(anom_scores[result["if_is_anomaly"] == 0], bins=40,
             alpha=0.6, color="#2ecc71", label="Normal")
    ax2.hist(anom_scores[result["if_is_anomaly"] == 1], bins=40,
             alpha=0.8, color="#e74c3c", label="Anomaly")
    ax2.set_title("Isolation Forest Anomaly Score")
    ax2.set_xlabel("Score (0–100, higher = more anomalous)")
    ax2.legend()

    # [3] Feature Importance (via mean absolute deviation in anomaly vs normal)
    ax3 = fig.add_subplot(gs[1, 1])
    available = [c for c in feature_cols if c in df_clean.columns]
    if "if_is_anomaly" in result.columns:
        merged = df_clean[available].copy()
        merged["is_anomaly"] = result.loc[df_clean.index, "if_is_anomaly"].values
        diff = (merged.groupby("is_anomaly")[available].mean().diff().iloc[-1].abs()
                .sort_values(ascending=True).tail(10))
        colors_fi = ["#e74c3c" if v > diff.quantile(0.7) else "#3498db" for v in diff.values]
        ax3.barh(diff.index, diff.values, color=colors_fi, edgecolor="white")
        ax3.set_title("Feature Deviation\nAnomaly vs Normal Days")
        ax3.set_xlabel("Mean Absolute Difference")

    plt.savefig(os.path.join(REPORT_DIR, "market_model_evaluation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log("Chart saved → reports/modelling/market_model_evaluation.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_market_modelling():
    print("\n" + "="*60)
    print("  MARKET RISK — Modelling & Early Warning")
    print("="*60)

    df, df_clean, feature_cols = load_data(INPUT_PATH)
    pipe, norm_scores, predictions = train_isolation_forest(df_clean, feature_cols)
    signals = detect_var_breaches(df)
    result  = generate_market_scores(df, df_clean, norm_scores, predictions, signals)

    plot_results(result, df_clean, feature_cols)
    result.to_csv(os.path.join(OUTPUT_DIR, "market_scores.csv"))

    joblib.dump(pipe, os.path.join(MODEL_DIR, "isolation_forest_market.pkl"))
    joblib.dump({"features": feature_cols, "trained_at": str(datetime.now())},
                os.path.join(MODEL_DIR, "market_metadata.pkl"))
    log("Model saved → models/market/isolation_forest_market.pkl")

    return pipe, result


if __name__ == "__main__":
    run_market_modelling()