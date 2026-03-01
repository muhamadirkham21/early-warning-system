"""
==============================================================
 Early Warning System — Master Feature Engineering Runner
 Jalankan semua domain sekaligus dari satu script
 
 Usage:
   python src/run_feature_engineering.py
==============================================================
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credit_risk.feature_engineering      import run_credit_feature_engineering
from market_risk.feature_engineering      import run_market_feature_engineering
from operational_risk.feature_engineering import run_operational_feature_engineering


def main():
    print("\n" + "█"*60)
    print("  EARLY WARNING SYSTEM — Fase 2: Feature Engineering")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█"*60)

    results = {}
    start_total = time.time()

    # ── 1. Credit Risk ──────────────────────────────────
    try:
        t = time.time()
        credit_df = run_credit_feature_engineering()
        results["Credit Risk"] = {
            "status": "✅ SUCCESS",
            "rows": len(credit_df),
            "features": credit_df.shape[1],
            "time": f"{time.time()-t:.1f}s"
        }
    except Exception as e:
        results["Credit Risk"] = {"status": f"❌ FAILED: {e}"}

    # ── 2. Market Risk ──────────────────────────────────
    try:
        t = time.time()
        market_df = run_market_feature_engineering()
        results["Market Risk"] = {
            "status": "✅ SUCCESS",
            "rows": len(market_df),
            "features": market_df.shape[1],
            "time": f"{time.time()-t:.1f}s"
        }
    except Exception as e:
        results["Market Risk"] = {"status": f"❌ FAILED: {e}"}

    # ── 3. Operational Risk ─────────────────────────────
    try:
        t = time.time()
        operational_df = run_operational_feature_engineering()
        results["Operational Risk"] = {
            "status": "✅ SUCCESS",
            "rows": len(operational_df),
            "features": operational_df.shape[1],
            "time": f"{time.time()-t:.1f}s"
        }
    except Exception as e:
        results["Operational Risk"] = {"status": f"❌ FAILED: {e}"}

    # ── Summary ─────────────────────────────────────────
    total_time = time.time() - start_total
    print("\n" + "="*60)
    print("  FASE 2 SUMMARY")
    print("="*60)
    for domain, info in results.items():
        print(f"\n  [{domain}]")
        for k, v in info.items():
            print(f"    {k:<12}: {v}")

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Completed : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n🎉 Fase 2 selesai! Data processed tersimpan di data/processed/")
    print("   Selanjutnya: Fase 3 — Modelling & Early Warning Logic\n")


if __name__ == "__main__":
    main()