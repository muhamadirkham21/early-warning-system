"""
==============================================================
 Early Warning System — Feature Engineering
 Domain: MARKET RISK
 Input : data/raw/market/combined_close_prices.csv
 Output: data/processed/market/market_features.csv
==============================================================

Fitur yang dibangun:
  1. Return & Volatility (Rolling)
  2. Drawdown Metrics
  3. Value at Risk (VaR) — Historical Simulation
  4. Technical Indicators (RSI, Bollinger Bands)
  5. Regime Detection (apakah pasar sedang "stress")
  6. Market Risk Score
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
INPUT_PATH  = os.path.join(BASE_DIR, "data", "raw", "market", "combined_close_prices.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed", "market")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "market_features.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAIN_TICKER   = "^JKSE"   # IHSG sebagai acuan utama
WINDOWS       = [5, 10, 20, 60]   # hari perdagangan (~1W, 2W, 1M, 3M)
VAR_CONF      = 0.95               # confidence level VaR

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ─────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df.dropna(how="all")
    log(f"✅ Loaded market data: {df.shape} | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────
# STEP 2: DAILY RETURNS
# ─────────────────────────────────────────
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    returns = df.pct_change()
    returns.columns = [f"ret_{col}" for col in df.columns]
    log(f"✅ Daily returns computed for {len(df.columns)} tickers")
    return returns


# ─────────────────────────────────────────
# STEP 3: ROLLING VOLATILITY
# Volatilitas = std dev return * sqrt(252) → annualized
# ─────────────────────────────────────────
def compute_rolling_volatility(returns: pd.DataFrame, ticker: str = MAIN_TICKER) -> pd.DataFrame:
    col = f"ret_{ticker}"
    if col not in returns.columns:
        col = returns.columns[0]  # fallback ke ticker pertama

    result = pd.DataFrame(index=returns.index)
    r = returns[col]

    for w in WINDOWS:
        result[f"vol_{w}d"]  = r.rolling(w).std() * np.sqrt(252)   # annualized
        result[f"ret_{w}d"]  = r.rolling(w).mean() * 252            # annualized mean return

    # Volatility regime: apakah volatilitas sekarang di atas rata-rata historisnya
    vol_20 = result["vol_20d"]
    result["vol_zscore"]     = (vol_20 - vol_20.rolling(252).mean()) / vol_20.rolling(252).std()
    result["is_high_vol"]    = (result["vol_zscore"] > 1.5).astype(int)

    log(f"✅ Rolling volatility computed (windows: {WINDOWS})")
    return result


# ─────────────────────────────────────────
# STEP 4: DRAWDOWN METRICS
# Max Drawdown: seberapa jauh harga turun dari puncaknya
# ─────────────────────────────────────────
def compute_drawdown(prices: pd.DataFrame, ticker: str = MAIN_TICKER) -> pd.DataFrame:
    col = ticker
    if col not in prices.columns:
        col = prices.columns[0]

    price = prices[col].dropna()
    result = pd.DataFrame(index=prices.index)

    # Rolling peak (cumulative max)
    rolling_max = price.cummax()

    # Drawdown setiap hari
    result["drawdown"]      = (price - rolling_max) / rolling_max
    result["drawdown_pct"]  = result["drawdown"] * 100

    # Rolling max drawdown (window 60 hari)
    result["max_drawdown_60d"] = result["drawdown"].rolling(60).min()

    # Flag: drawdown lebih dari 5% (threshold sinyal awal)
    result["flag_drawdown_5pct"]  = (result["drawdown"] < -0.05).astype(int)
    result["flag_drawdown_10pct"] = (result["drawdown"] < -0.10).astype(int)
    result["flag_drawdown_20pct"] = (result["drawdown"] < -0.20).astype(int)  # severe crash

    log("✅ Drawdown metrics computed")
    return result


# ─────────────────────────────────────────
# STEP 5: VALUE AT RISK (VaR) — Historical Simulation
# VaR: kerugian maksimum dalam 1 hari dengan confidence level tertentu
# ─────────────────────────────────────────
def compute_var(returns: pd.DataFrame, ticker: str = MAIN_TICKER, window: int = 252) -> pd.DataFrame:
    col = f"ret_{ticker}"
    if col not in returns.columns:
        col = returns.columns[0]

    r = returns[col]
    result = pd.DataFrame(index=returns.index)

    # Historical VaR: percentile dari distribusi return historis
    result["var_95_1d"] = r.rolling(window).quantile(1 - VAR_CONF)
    result["var_99_1d"] = r.rolling(window).quantile(0.01)

    # Expected Shortfall (CVaR): rata-rata kerugian di atas VaR
    def cvar(x):
        threshold = np.percentile(x.dropna(), (1 - VAR_CONF) * 100)
        tail = x[x <= threshold]
        return tail.mean() if len(tail) > 0 else np.nan

    result["cvar_95_1d"] = r.rolling(window).apply(cvar, raw=False)

    # Flag: return harian lebih buruk dari VaR (VaR breach)
    result["var_breach_95"] = (r < result["var_95_1d"]).astype(int)
    result["var_breach_99"] = (r < result["var_99_1d"]).astype(int)

    log("✅ VaR & CVaR computed (95% & 99%)")
    return result


# ─────────────────────────────────────────
# STEP 6: TECHNICAL INDICATORS
# ─────────────────────────────────────────
def compute_technical_indicators(prices: pd.DataFrame, ticker: str = MAIN_TICKER) -> pd.DataFrame:
    col = ticker
    if col not in prices.columns:
        col = prices.columns[0]

    price = prices[col]
    result = pd.DataFrame(index=prices.index)

    # ── RSI (Relative Strength Index) ──
    # RSI < 30: oversold (potensi rebound), RSI > 70: overbought (potensi koreksi)
    delta = price.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-9)
    result["rsi_14"] = 100 - (100 / (1 + rs))
    result["flag_oversold"]   = (result["rsi_14"] < 30).astype(int)
    result["flag_overbought"] = (result["rsi_14"] > 70).astype(int)

    # ── Bollinger Bands ──
    # Harga menembus band bawah = sinyal stress pasar
    sma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    result["bb_upper"] = sma20 + (2 * std20)
    result["bb_lower"] = sma20 - (2 * std20)
    result["bb_width"]  = (result["bb_upper"] - result["bb_lower"]) / sma20  # volatility proxy
    result["flag_bb_breach_lower"] = (price < result["bb_lower"]).astype(int)

    # ── Moving Average Cross (trend) ──
    result["sma_20"] = sma20
    result["sma_60"] = price.rolling(60).mean()
    result["trend_bearish"] = (result["sma_20"] < result["sma_60"]).astype(int)  # death cross

    log("✅ Technical indicators computed (RSI, Bollinger Bands, MA Cross)")
    return result


# ─────────────────────────────────────────
# STEP 7: MARKET RISK SCORE
# ─────────────────────────────────────────
def build_market_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    # Komponen skor
    vol_score   = minmax(df.get("vol_20d", pd.Series(0, index=df.index)).fillna(0))
    dd_score    = minmax(df.get("drawdown_pct", pd.Series(0, index=df.index)).fillna(0).abs())
    var_score   = minmax(df.get("var_95_1d", pd.Series(0, index=df.index)).fillna(0).abs())
    
    flag_cols   = [c for c in df.columns if c.startswith("flag_")]
    flag_score  = df[flag_cols].sum(axis=1) / max(len(flag_cols), 1) if flag_cols else pd.Series(0, index=df.index)

    df["market_risk_score"] = (
        vol_score  * 0.35 +
        dd_score   * 0.30 +
        var_score  * 0.20 +
        flag_score * 0.15
    ) * 100

    df["market_risk_label"] = pd.cut(
        df["market_risk_score"],
        bins=[0, 33, 66, 100],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True
    )

    log("✅ Market risk score built")
    log(f"   Distribution:\n{df['market_risk_label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_market_feature_engineering():
    print("\n" + "="*60)
    print("  MARKET RISK — Feature Engineering")
    print("="*60)

    prices  = load_data(INPUT_PATH)
    returns = compute_returns(prices)

    volatility  = compute_rolling_volatility(returns)
    drawdown    = compute_drawdown(prices)
    var_metrics = compute_var(returns)
    technicals  = compute_technical_indicators(prices)

    # Gabungkan semua fitur
    df = pd.concat([prices, volatility, drawdown, var_metrics, technicals], axis=1)
    df = build_market_risk_score(df)

    df.dropna(subset=["vol_20d"], inplace=True)  # hapus baris awal yang belum ada rolling value
    df.to_csv(OUTPUT_PATH)

    log(f"\n💾 Output saved → {OUTPUT_PATH}")
    log(f"   Shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = run_market_feature_engineering()
    print("\n📊 Sample output (last 5 rows):")
    cols_show = ["vol_20d", "drawdown_pct", "var_95_1d", "rsi_14",
                 "market_risk_score", "market_risk_label"]
    show = [c for c in cols_show if c in df.columns]
    print(df[show].tail(5).to_string())