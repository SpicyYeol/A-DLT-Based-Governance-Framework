# -*- coding: utf-8 -*-
"""How much of the headline detection result rests on S-DF-03, whose label
distribution P_a is computed from the raw local labels in the testbed and
would have to be self-reported (or inferred) in a deployment?

Conditions, primary profile, seeds 42-46:
  full            all 36 controls
  no_sdf03        S-DF-03 disabled (no trusted label-distribution channel)
  no_stat_privy   S-DF-03 disabled AND M-BC-01 kept (update-norm only)
Usage: python sdf03.py <out.json>
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                  # noqa: E402
from trust import fl_simulation as fl                      # noqa: E402
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402
from trust.taxonomy import ALL_CONTROLS                    # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
CONDITIONS = {
    "full": frozenset(),
    "no_sdf03": frozenset({"S-DF-03"}),
}


def metrics(results, ids, truth, disabled):
    flagged = {a for a in ids
               if agent_verdict(results, a, disabled=disabled)[0] != "Trusted"}
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    tn = len(set(ids) - truth - flagged)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 1.0}


def main(out_path):
    truth = set(config.ADVERSARY_PROFILE) | {config.REVOKED_AGENT}
    per_cond = {k: [] for k in CONDITIONS}
    missed = {k: [] for k in CONDITIONS}
    for s in SEEDS:
        prep = fl.prepare(s)
        tr = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                 profile=config.ADVERSARY_PROFILE)
        ctx = build_ctx(tr["signals"])
        res = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        for cname, dis in CONDITIONS.items():
            m = metrics(res, ids, truth, dis)
            per_cond[cname].append(m)
            flagged = {a for a in ids
                       if agent_verdict(res, a, disabled=dis)[0] != "Trusted"}
            for a in sorted(truth - flagged):
                missed[cname].append({"seed": s, "agent": a,
                                      "behaviour": ctx.agents[a].get("behaviour")})
        print("  seed %d done" % s, flush=True)

    out = {"seeds": SEEDS, "truth": sorted(truth), "conditions": {}}
    for cname in CONDITIONS:
        rows = per_cond[cname]
        out["conditions"][cname] = {
            "recall_mean": round(st.mean(r["recall"] for r in rows), 4),
            "recall_sd": round(st.pstdev([r["recall"] for r in rows]), 4),
            "precision_mean": round(st.mean(r["precision"] for r in rows), 4),
            "TP": sum(r["TP"] for r in rows), "FN": sum(r["FN"] for r in rows),
            "FP": sum(r["FP"] for r in rows), "TN": sum(r["TN"] for r in rows),
            "per_seed_recall": [r["recall"] for r in rows],
            "missed": missed[cname],
        }
        c = out["conditions"][cname]
        print("  [%-9s] recall %.4f+-%.4f  TP %d FN %d  missed=%s"
              % (cname, c["recall_mean"], c["recall_sd"], c["TP"], c["FN"],
                 sorted({m["behaviour"] for m in c["missed"]})), flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1, default=str),
                              encoding="utf-8")
    print("saved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
