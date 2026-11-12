# -*- coding: utf-8 -*-
"""Attribution versus mitigation over a realistic round horizon.

Robust aggregation dilutes an adversarial update every round but never removes
the participant, so the same violator keeps entering the aggregate. Governance
adjudicates once and excludes, so the violator stops contributing.

For each condition we record, per round, which participants' updates actually
entered the global model, and from that:
  rounds_with_adversary   rounds in which at least one adversarial update was
                          aggregated
  adv_updates_admitted    total adversarial updates aggregated over the run
  adv_share               mean share of the aggregated set that was adversarial

Usage: python persist.py <out.json> [rounds]
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
from trust import aggregators as agg                        # noqa: E402
from trust import fl_simulation as fl                       # noqa: E402
from trust.engine import build_ctx, evaluate_controls, score  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
F = 2


def flat(c):
    return agg._flatten(c)


def make_recorder(kind, log):
    """Return an aggregator that records which aids entered the aggregate."""
    def fedavg_rec(contribs):
        log.append([c["aid"] for c in contribs])
        return agg.fedavg(contribs)

    def trimmed_rec(contribs):
        # coordinate-wise trimming keeps every participant in the input set;
        # there is no participant-level exclusion
        log.append([c["aid"] for c in contribs])
        return agg.trimmed_mean(contribs)

    def median_rec(contribs):
        log.append([c["aid"] for c in contribs])
        return agg.coordinate_median(contribs)

    def krum_rec(contribs):
        M = np.asarray([flat(c) for c in contribs])
        if len(M) <= F + 2:
            log.append([c["aid"] for c in contribs])
            return agg.fedavg(contribs)
        scores = agg._krum_scores(M, F)
        pick = int(np.argmin(scores))
        log.append([contribs[pick]["aid"]])
        return agg._unflatten(M[pick], contribs[0])

    def mkrum_rec(contribs):
        M = np.asarray([flat(c) for c in contribs])
        n = len(M)
        if n <= F + 2:
            log.append([c["aid"] for c in contribs])
            return agg.fedavg(contribs)
        scores = agg._krum_scores(M, F)
        m = max(1, n - F)
        chosen = np.argsort(scores)[:m]
        log.append([contribs[int(i)]["aid"] for i in chosen])
        return agg._unflatten(M[chosen].mean(axis=0), contribs[0])

    return {"FedAvg": fedavg_rec, "Trimmed-Mean": trimmed_rec,
            "Coord-Median": median_rec, "Krum": krum_rec,
            "Multi-Krum": mkrum_rec}[kind]


def summarise(log, adversaries, rounds):
    rounds_hit = sum(1 for sel in log if any(a in adversaries for a in sel))
    admitted = sum(sum(1 for a in sel if a in adversaries) for sel in log)
    shares = [sum(1 for a in sel if a in adversaries) / max(len(sel), 1)
              for sel in log]
    return {"rounds": len(log), "rounds_with_adversary": rounds_hit,
            "adv_updates_admitted": admitted,
            "adv_share": round(st.mean(shares), 4) if shares else 0.0}


def main(out_path):
    profile = config.ADVERSARY_PROFILE
    truth = set(profile) | {config.REVOKED_AGENT}
    old_rounds = config.NUM_ROUNDS
    config.NUM_ROUNDS = ROUNDS
    per_cond = {}
    try:
        for s in SEEDS:
            prep = fl.prepare(s)

            # governance: adjudicate from recorded evidence, then exclude
            rec = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                      profile=profile)
            ctx = build_ctx(rec["signals"])
            sc = score(ctx, evaluate_controls(ctx))
            flagged = {a for a, v in sc["agents"].items()
                       if v["verdict"] != "Trusted"}
            glog = []
            gacc = fl.train_trajectory(
                prep, exclude=frozenset(flagged), profile=profile,
                aggregator=make_recorder("FedAvg", glog))["accuracy"]
            g = summarise(glog, truth, ROUNDS)
            g["accuracy"] = gacc
            g["flagged"] = sorted(flagged)
            per_cond.setdefault("Governance + FedAvg", []).append(g)

            for kind in ("FedAvg", "Krum", "Multi-Krum",
                         "Trimmed-Mean", "Coord-Median"):
                log = []
                acc = fl.train_trajectory(
                    prep, profile=profile,
                    aggregator=make_recorder(kind, log))["accuracy"]
                r = summarise(log, truth, ROUNDS)
                r["accuracy"] = acc
                name = kind + " alone" if kind != "FedAvg" else "No defense (FedAvg)"
                per_cond.setdefault(name, []).append(r)
            print("  seed %d done (flagged %s)" % (s, sorted(flagged)), flush=True)
    finally:
        config.NUM_ROUNDS = old_rounds

    order = ["No defense (FedAvg)", "Krum alone", "Multi-Krum alone",
             "Trimmed-Mean alone", "Coord-Median alone", "Governance + FedAvg"]
    summary = {}
    print()
    print("%-22s %8s %10s %10s %9s" % ("condition", "acc", "rounds/adv",
                                       "adv admit", "adv share"))
    for name in order:
        rows = per_cond[name]
        summary[name] = {
            "accuracy_mean": round(st.mean(r["accuracy"] for r in rows), 4),
            "accuracy_sd": round(st.pstdev([r["accuracy"] for r in rows]), 4),
            "rounds_with_adversary_mean": round(
                st.mean(r["rounds_with_adversary"] for r in rows), 2),
            "adv_updates_admitted_mean": round(
                st.mean(r["adv_updates_admitted"] for r in rows), 1),
            "adv_share_mean": round(st.mean(r["adv_share"] for r in rows), 4),
        }
        d = summary[name]
        print("%-22s %8.4f %6.1f/%-3d %10.1f %9.3f"
              % (name, d["accuracy_mean"], d["rounds_with_adversary_mean"],
                 ROUNDS, d["adv_updates_admitted_mean"], d["adv_share_mean"]))

    Path(out_path).write_text(json.dumps(
        {"rounds": ROUNDS, "seeds": SEEDS, "adversaries": sorted(truth),
         "summary": summary, "per_seed": per_cond}, indent=1), encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
