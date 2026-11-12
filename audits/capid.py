# -*- coding: utf-8 -*-
"""Identify which TRUST_IMAGE_CAP reproduces the submitted primary figures
(naive 0.1792 +/- 0.0323, governance 0.8506 +/- 0.0104, seeds 42-46).
Usage: python capid.py <out.json>
"""
from __future__ import annotations


import json
import os
import statistics as st
import sys
from pathlib import Path

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp12"))
sys.path.insert(0, str(TB / "src"))

SEEDS = [42, 43, 44, 45, 46]
CAPS = [2000, 4000, 8000, 16000, 70000]
TARGET_NAIVE, TARGET_GOV = 0.1792, 0.8506


def run_cap(cap):
    from trust import config
    config.IMAGE_SAMPLE_CAP = cap
    from trust import fl_simulation as fl
    from trust.engine import build_ctx, evaluate_controls, score

    naive, gov, nper = [], [], None
    for s in SEEDS:
        prep = fl.prepare(s)
        if nper is None:
            nper = round(len(prep["X_tr"]) / config.NUM_AGENTS, 1)
        tr = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                 profile=config.ADVERSARY_PROFILE)
        ctx = build_ctx(tr["signals"])
        res = evaluate_controls(ctx)
        sc = score(ctx, res)
        bad = {a for a, v in sc["agents"].items() if v["verdict"] != "Trusted"}
        gated = fl.train_trajectory(prep, exclude=frozenset(bad),
                                    profile=config.ADVERSARY_PROFILE)
        naive.append(tr["accuracy"])
        gov.append(gated["accuracy"])
    return naive, gov, nper


def main(out_path):
    rows = []
    for cap in CAPS:
        naive, gov, nper = run_cap(cap)
        row = {"cap": cap,
               "naive_mean": round(st.mean(naive), 4),
               "naive_std": round(st.pstdev(naive), 4),
               "gov_mean": round(st.mean(gov), 4),
               "gov_std": round(st.pstdev(gov), 4),
               "samples_per_participant": nper,
               "naive_per_seed": [round(x, 4) for x in naive],
               "gov_per_seed": [round(x, 4) for x in gov]}
        row["dist_to_submitted"] = round(
            abs(row["naive_mean"] - TARGET_NAIVE)
            + abs(row["gov_mean"] - TARGET_GOV), 4)
        rows.append(row)
        print("  cap=%-6d naive %.4f+-%.4f  gov %.4f+-%.4f   |dist|=%.4f"
              % (cap, row["naive_mean"], row["naive_std"],
                 row["gov_mean"], row["gov_std"], row["dist_to_submitted"]),
              flush=True)
    best = min(rows, key=lambda r: r["dist_to_submitted"])
    print("\nclosest to submitted (naive %.4f / gov %.4f): cap=%d"
          % (TARGET_NAIVE, TARGET_GOV, best["cap"]))
    Path(out_path).write_text(
        json.dumps({"target": {"naive": TARGET_NAIVE, "gov": TARGET_GOV},
                    "seeds": SEEDS, "rows": rows, "closest_cap": best["cap"]},
                   indent=1), encoding="utf-8")
    print("saved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
