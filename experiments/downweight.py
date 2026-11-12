# -*- coding: utf-8 -*-
"""R4(4): evaluate down-weighting of Under Review contributions as an
alternative to quarantining them for the current round.

Conditions (primary profile, 5 seeds, cap 4000):
  no_action            every contribution aggregated
  quarantine           Untrusted excluded AND Under Review excluded  (paper policy)
  downweight_L         Untrusted excluded, Under Review weight x L
  ground_truth         the injected adversaries excluded

Usage: python downweight.py <out.json>
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
LAMBDAS = [0.25, 0.50, 0.75]


def weighted(contribs, scale):
    """FedAvg with a per-agent weight multiplier."""
    w = np.asarray([c["weight"] * scale.get(c["aid"], 1.0) for c in contribs],
                   dtype=float)
    tot = w.sum()
    if tot <= 0:
        w = np.asarray([c["weight"] for c in contribs], dtype=float)
        tot = w.sum()
    w = w / tot
    coef = np.tensordot(w, np.asarray([c["coef"] for c in contribs]), axes=(0, 0))
    intc = np.tensordot(w, np.asarray([c["int"] for c in contribs]), axes=(0, 0))
    return coef, intc


def main(out_path):
    profile = config.ADVERSARY_PROFILE
    gt_bad = set(profile) | {config.REVOKED_AGENT}
    rows = []
    for s in SEEDS:
        prep = fl.prepare(s)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                    profile=profile)
        ctx = build_ctx(naive["signals"])
        res = evaluate_controls(ctx)
        sc = score(ctx, res)
        untrusted = {a for a, v in sc["agents"].items() if v["verdict"] == "Untrusted"}
        review = {a for a, v in sc["agents"].items() if v["verdict"] == "Under Review"}

        r = {"seed": s,
             "untrusted": sorted(untrusted), "under_review": sorted(review),
             "no_action": naive["accuracy"],
             "quarantine": fl.train_trajectory(
                 prep, exclude=frozenset(untrusted | review),
                 profile=profile)["accuracy"],
             "ground_truth": fl.train_trajectory(
                 prep, exclude=frozenset(gt_bad), profile=profile)["accuracy"]}
        for lam in LAMBDAS:
            scale = {a: lam for a in review}
            r["downweight_%.2f" % lam] = fl.train_trajectory(
                prep, exclude=frozenset(untrusted), profile=profile,
                aggregator=lambda c, _s=scale: weighted(c, _s))["accuracy"]
        rows.append(r)
        print("  seed %d  untrusted=%s under_review=%s" % (s, r["untrusted"],
                                                           r["under_review"]),
              flush=True)

    keys = (["no_action", "quarantine"]
            + ["downweight_%.2f" % l for l in LAMBDAS] + ["ground_truth"])
    summary = {k: {"mean": round(st.mean(r[k] for r in rows), 4),
                   "sd": round(st.pstdev([r[k] for r in rows]), 4)} for k in keys}
    print()
    for k in keys:
        print("  %-16s %.4f +- %.4f" % (k, summary[k]["mean"], summary[k]["sd"]))
    Path(out_path).write_text(
        json.dumps({"seeds": SEEDS, "lambdas": LAMBDAS,
                    "summary": summary, "rows": rows}, indent=1),
        encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
