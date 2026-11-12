# -*- coding: utf-8 -*-
"""Revision 1 (IEEE Access) — R2-4: detection over an extended seed set.

Primary threat profile, seeds 42..71 (30 seeds). Records per-seed TP/FP/FN/TN
so that exact (Clopper-Pearson) intervals can be computed on the pooled counts.
Run:  TRUST_DATA_ROOT=<dir> python scripts/rev1_seeds.py
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trust import experiments as E, config

OUT = config.DATA_ROOT / "reports_rev1"
OUT.mkdir(parents=True, exist_ok=True)
seeds = list(range(42, 72))
rows = []
for s in seeds:
    t = time.perf_counter()
    r = E.run_seed(s, ablations=False)
    d = r["detection"]
    rows.append({"seed": s, "naive": r["naive_accuracy"], "gov": r["governance_accuracy"],
                 "oracle": r["oracle_accuracy"], **{k: d[k] for k in ("TP", "FP", "FN", "TN")},
                 "flagged": r["flagged_agents"],
                 "verdicts": {a: v["verdict"] for a, v in r["verdicts"].items()},
                 "sec": round(time.perf_counter() - t, 1)})
    print(json.dumps(rows[-1]), flush=True)
json.dump({"profile": "primary", "seeds": seeds, "rows": rows},
          open(OUT / "rev1_seeds.json", "w"), indent=1)
