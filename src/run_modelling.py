"""
==============================================================
 Early Warning System — Master Modelling Runner
 Jalankan semua model + generate unified EWS report

 Usage:
   python src/run_modelling.py
==============================================================
"""

import sys, os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credit_risk.modelling      import run_credit_modelling
from market_risk.modelling      import run_market_modelling
from operational_risk.modelling import run_operational_modelling

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "reports", "modelling")
os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────
# UNIFIED EWS SUMMARY REPORT
# ─────────────────────────────────────────
def generate_ews_summary():
    """
    Baca hasil scoring dari ketiga domain dan buat
    satu ringkasan EWS terpadu yang siap dibawa ke dashboard.
    """
    print("\n" + "─"*60)
    print("  Generating Unified EWS Summary...")
    print("─"*60)

    summary = {}

    # ── Credit ──────────────────────────────────────────────
    try:
        credit_path = os.path.join(BASE_DIR, "data", "processed", "credit", "credit_scores.csv")
        credit = pd.read_csv(credit_path)
        summary["credit"] = {
            "total":    len(credit),
            "NORMAL":   (credit["ews_signal"] == "NORMAL").sum(),
            "WARNING":  (credit["ews_signal"] == "WARNING").sum(),
            "CRITICAL": (credit["ews_signal"] == "CRITICAL").sum(),
            "avg_score": credit["ml_risk_score"].mean(),
        }
    except Exception as e:
        summary["credit"] = {"error": str(e)}

    # ── Market ──────────────────────────────────────────────
    try:
        market_path = os.path.join(BASE_DIR, "data", "processed", "market", "market_scores.csv")
        market = pd.read_csv(market_path, index_col=0, parse_dates=True)
        last_signal = market["ews_signal"].dropna().iloc[-1]
        last_score  = market["market_ews_score"].dropna().iloc[-1]
        summary["market"] = {
            "total":         len(market),
            "NORMAL":        (market["ews_signal"] == "NORMAL").sum(),
            "WARNING":       (market["ews_signal"] == "WARNING").sum(),
            "CRITICAL":      (market["ews_signal"] == "CRITICAL").sum(),
            "latest_signal": last_signal,
            "latest_score":  round(last_score, 1),
        }
    except Exception as e:
        summary["market"] = {"error": str(e)}

    # ── Operational ─────────────────────────────────────────
    try:
        ops_path = os.path.join(BASE_DIR, "data", "processed", "operational", "operational_scores.csv")
        ops = pd.read_csv(ops_path)
        summary["operational"] = {
            "total":          len(ops),
            "NORMAL":         (ops["ews_signal"] == "NORMAL").sum(),
            "WARNING":        (ops["ews_signal"] == "WARNING").sum(),
            "CRITICAL":       (ops["ews_signal"] == "CRITICAL").sum(),
            "avg_fraud_prob": ops["ml_fraud_prob"].mean(),
        }
    except Exception as e:
        summary["operational"] = {"error": str(e)}

    return summary


def print_ews_report(summary: dict, model_results: dict):
    report = f"""
{'█'*65}
  EARLY WARNING SYSTEM — UNIFIED RISK REPORT
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'█'*65}

┌─────────────────────────────────────────────────────────────┐
│                    MODEL PERFORMANCE                         │
├──────────────────────┬──────────────────┬───────────────────┤
│ Domain               │ Metric           │ Score             │
├──────────────────────┼──────────────────┼───────────────────┤
│ Credit Risk          │ AUC-ROC          │ {model_results.get('credit_auc', 'N/A'):<17} │
│ Market Risk          │ Anomaly %        │ {model_results.get('market_anomaly', 'N/A'):<17} │
│ Operational Risk     │ AUC-ROC          │ {model_results.get('ops_auc', 'N/A'):<17} │
└──────────────────────┴──────────────────┴───────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  EWS SIGNAL DISTRIBUTION                     │
├──────────────────┬──────────┬───────────┬───────────────────┤
│ Domain           │ NORMAL   │ WARNING   │ CRITICAL          │
├──────────────────┼──────────┼───────────┼───────────────────┤"""

    for domain, info in summary.items():
        if "error" not in info:
            n = info.get("NORMAL",   0)
            w = info.get("WARNING",  0)
            c = info.get("CRITICAL", 0)
            report += f"\n│ {domain.capitalize():<16} │ {n:<8} │ {w:<9} │ {c:<17} │"

    report += f"""
└──────────────────┴──────────┴───────────┴───────────────────┘

LATEST MARKET SIGNAL : {summary.get('market', {}).get('latest_signal', 'N/A')} 
                       (Score: {summary.get('market', {}).get('latest_score', 'N/A')})

{'█'*65}
"""
    print(report)

    # Simpan ke file
    report_path = os.path.join(BASE_DIR, "reports", "ews_unified_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  📄 Report saved → {report_path}")


def plot_ews_overview(summary: dict):
    """Traffic light dashboard sederhana untuk semua domain."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Early Warning System — Unified Risk Dashboard",
                 fontsize=15, fontweight="bold")

    colors = {"NORMAL": "#2ecc71", "WARNING": "#f39c12", "CRITICAL": "#e74c3c"}
    domains = [
        ("credit",      "💳 Credit Risk",      "Debitur"),
        ("market",      "📈 Market Risk",       "Hari Trading"),
        ("operational", "🔐 Operational Risk",  "Transaksi"),
    ]

    for ax, (key, title, unit) in zip(axes, domains):
        info = summary.get(key, {})
        if "error" in info:
            ax.text(0.5, 0.5, f"Error:\n{info['error']}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
            ax.set_title(title)
            continue

        vals  = [info.get("NORMAL",0), info.get("WARNING",0), info.get("CRITICAL",0)]
        labs  = ["NORMAL", "WARNING", "CRITICAL"]
        cols  = [colors[l] for l in labs]
        total = sum(vals)

        wedges, texts, autotexts = ax.pie(
            vals, labels=labs, autopct="%1.1f%%",
            colors=cols, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 9}
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_fontweight("bold")

        ax.set_title(f"{title}\n(n={total:,} {unit})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "ews_unified_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Chart saved → reports/modelling/ews_unified_overview.png")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("\n" + "█"*65)
    print("  EARLY WARNING SYSTEM — Fase 3: Modelling")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*65)

    model_results = {}
    start = time.time()

    # ── 1. Credit Risk ──────────────────────────────────────
    try:
        t = time.time()
        _, threshold, auc = run_credit_modelling()
        model_results["credit_auc"]       = f"{auc:.4f}"
        model_results["credit_threshold"] = f"{threshold:.3f}"
        print(f"\n  ✅ Credit Risk done ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"\n  ❌ Credit Risk FAILED: {e}")
        model_results["credit_auc"] = "FAILED"

    # ── 2. Market Risk ─────────────────────────────────────
    try:
        t = time.time()
        _, market_result = run_market_modelling()
        anomaly_pct = (market_result["if_is_anomaly"] == 1).mean()
        model_results["market_anomaly"] = f"{anomaly_pct:.1%} anomaly days"
        print(f"\n  ✅ Market Risk done ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"\n  ❌ Market Risk FAILED: {e}")
        model_results["market_anomaly"] = "FAILED"

    # ── 3. Operational Risk ────────────────────────────────
    try:
        t = time.time()
        _, threshold, auc = run_operational_modelling()
        model_results["ops_auc"]       = f"{auc:.4f}"
        model_results["ops_threshold"] = f"{threshold:.3f}"
        print(f"\n  ✅ Operational Risk done ({time.time()-t:.1f}s)")
    except Exception as e:
        print(f"\n  ❌ Operational Risk FAILED: {e}")
        model_results["ops_auc"] = "FAILED"

    # ── 4. Unified Report ──────────────────────────────────
    summary = generate_ews_summary()
    print_ews_report(summary, model_results)
    plot_ews_overview(summary)

    total = time.time() - start
    print(f"\n🎉 Fase 3 selesai! Total waktu: {total:.1f}s")
    print(f"   Model tersimpan di: models/")
    print(f"   Report tersimpan di: reports/")
    print(f"\n   Selanjutnya → Fase 4: Dashboard & Alert System\n")


if __name__ == "__main__":
    main()