# -*- coding: utf-8 -*-
"""Audit the claimed per-agent aggregation weight cap of 0.30.

The experimental-setup table (Table 5 of the paper) stated "FedAvg weighted by
local sample count; per-agent weight
limited to 0.30". The aggregation helper `_wavg` normalises by sample count
and applies no cap, so after governance leaves three survivors at least one
weight must exceed 1/3. This prints the actual normalised weights, and the
separate influence score that M-SC-02 compares against 0.30.

Usage: python weightcheck.py <out.json>
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                   # noqa: E402
from trust import fl_simulation as fl                       # noqa: E402
from trust.engine import build_ctx, evaluate_controls, score  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]


def main(out_path):
    profile = config.ADVERSARY_PROFILE
    rows = []
    for s in SEEDS:
        prep = fl.prepare(s)
        rec = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                  profile=profile)
        ctx = build_ctx(rec["signals"])
        sc = score(ctx, evaluate_controls(ctx))
        flagged = {a for a, v in sc["agents"].items() if v["verdict"] != "Trusted"}

        shards = {sh["agent_id"]: len(sh["y"]) for sh in prep["shards"]}
        survivors = [a for a in sorted(shards) if a not in flagged]
        n = np.array([shards[a] for a in survivors], float)
        w = n / n.sum()

        # influence score as M-SC-02 computes it, over ALL agents
        ids = ctx.agent_ids()
        norms = np.array([st.mean([r["update_norm"]
                                   for r in ctx.agents[i]["rounds"]])
                          for i in ids])
        samples = np.array([ctx.agents[i]["n_samples"] for i in ids], float)
        sw = samples / samples.sum()
        infl = sw * norms
        infl = infl / max(infl.sum(), 1e-9)

        rows.append({
            "seed": s,
            "excluded": sorted(flagged),
            "survivors": survivors,
            "survivor_samples": [int(shards[a]) for a in survivors],
            "survivor_agg_weights": [round(float(x), 4) for x in w],
            "max_agg_weight": round(float(w.max()), 4),
            "influence_all_agents": {str(i): round(float(v), 4)
                                     for i, v in zip(ids, infl)},
            "influence_over_0.30": sorted(int(i) for i, v in zip(ids, infl)
                                          if v > 0.30)})
        print("  seed %d survivors=%s agg weights=%s (max %.4f)"
              % (s, survivors, rows[-1]["survivor_agg_weights"],
                 rows[-1]["max_agg_weight"]), flush=True)

    mx = [r["max_agg_weight"] for r in rows]
    print()
    print("maximum post-governance aggregation weight: %.4f to %.4f (mean %.4f)"
          % (min(mx), max(mx), st.mean(mx)))
    print("claimed cap in the setup table: 0.30  ->  violated in %d of %d seeds"
          % (sum(1 for v in mx if v > 0.30), len(mx)))
    print("lower bound with three survivors: 1/3 = 0.3333")
    print()
    print("influence scores above 0.30 (what M-SC-02 actually tests):")
    for r in rows:
        print("  seed %d: %s" % (r["seed"], r["influence_over_0.30"]))

    Path(out_path).write_text(json.dumps({"seeds": SEEDS, "rows": rows},
                                         indent=1), encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
