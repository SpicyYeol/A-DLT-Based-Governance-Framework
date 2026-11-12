"""Run the full governance pipeline locally, without Airflow.

Same logic the DAG executes, useful for development and CI. Writes reports to
$TRUST_DATA_ROOT/reports (defaults to ./data/reports when run from repo root).

    python scripts/run_local.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# make the trust package importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TRUST_DATA_ROOT",
                      str(Path(__file__).resolve().parents[1] / "data"))


def main() -> int:
    from trust.fl_simulation import run_simulation
    from trust.tabular_security import run_detection
    from trust.engine import run_governance
    from trust import report as rpt

    print("[1/4] Federated-learning simulation (image dataset) ...")
    signals = run_simulation(reset_ledger=True)
    print(f"      dataset={signals['dataset']}  "
          f"naive_acc={signals['global_accuracy']:.4f}  "
          f"gated_acc={signals['gated_accuracy']:.4f}  "
          f"(uplift {signals['accuracy_uplift_from_gating']:+.4f})")

    print("[2/4] Tabular security detection ...")
    tab = run_detection()
    print(f"      dataset={tab['dataset']}  "
          f"pred_anomaly_rate={tab['anomaly_rate_pred']:.3f}")

    print("[3/4] Evaluating 36 controls + scoring ...")
    report = run_governance(signals)
    report["tabular_security"] = tab

    print("[4/4] Writing reports ...")
    csv1 = rpt.write_structure_trust_factors(report)
    csv2 = rpt.write_agent_verdicts(report)
    js = rpt.write_summary_json(report)

    print()
    print(rpt.render_markdown_summary(report))
    print()
    print(f"Structure Trust Factors (filled): {csv1}")
    print(f"Agent verdicts:                   {csv2}")
    print(f"Full JSON report:                 {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
