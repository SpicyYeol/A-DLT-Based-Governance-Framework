# -*- coding: utf-8 -*-
"""Governance overhead against the number of training rounds.

Repeats the primary protocol at 3, 10, 25 and 50 rounds and measures both the
governance cost as a share of training time and the stability of detection, so
the three-round figure can be read against a longer horizon. Reported as the
round-count table (Table S5 of the paper).

Usage:  python exp3.py <out.json>
"""
from __future__ import annotations
import os, sys, json, time, statistics as st
from pathlib import Path

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config
from trust import fl_simulation as fl
from trust.engine import build_ctx, evaluate_controls
from trust.ledger import Ledger
from trust.experiments import _flagged_set, _conf

ROUNDS = [3, 10, 25, 50]
SEEDS = [42, 43, 44, 45, 46]


def measure(seed: int) -> dict:
    profile = config.ADVERSARY_PROFILE
    prep = fl.prepare(seed)
    t0 = time.perf_counter()
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    t_fl = time.perf_counter() - t0

    t0 = time.perf_counter()
    ctx = build_ctx(naive["signals"])
    res = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    flagged = _flagged_set(res, ids)
    t_gov = time.perf_counter() - t0

    led = Ledger()
    t0 = time.perf_counter()
    chain = led.verify_chain()
    t_ver = time.perf_counter() - t0
    nbytes = led.path.stat().st_size if led.path.exists() else 0

    gt = set(profile) | {config.REVOKED_AGENT}
    det = _conf(flagged, ids, gt)
    gov_acc = fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"]
    return {"t_fl_s": round(t_fl, 3), "t_gov_s": round(t_gov, 4),
            "t_verify_s": round(t_ver, 4), "ledger_blocks": chain["blocks"],
            "ledger_bytes": nbytes, "naive": round(naive["accuracy"], 4),
            "gov": round(gov_acc, 4), **{k: det[k] for k in
            ("TP", "FP", "FN", "TN", "recall", "false_positive_rate")}}


def main(out: str) -> int:
    saved = config.NUM_ROUNDS
    rows = []
    try:
        for nr in ROUNDS:
            config.NUM_ROUNDS = nr
            per = []
            for s in SEEDS:
                per.append(measure(s))
                print(f"  rounds={nr:3d} seed={s}: gov_t={per[-1]['t_gov_s']:.3f}s "
                      f"fl_t={per[-1]['t_fl_s']:.2f}s blocks={per[-1]['ledger_blocks']} "
                      f"recall={per[-1]['recall']} fpr={per[-1]['false_positive_rate']}",
                      flush=True)
            def m(k):
                v = [p[k] for p in per]
                return round(st.mean(v), 4)
            TP = sum(p["TP"] for p in per); FP = sum(p["FP"] for p in per)
            FN = sum(p["FN"] for p in per); TN = sum(p["TN"] for p in per)
            rows.append({"rounds": nr, "n_seeds": len(SEEDS),
                         "t_fl_s": m("t_fl_s"), "t_gov_s": m("t_gov_s"),
                         "t_verify_s": m("t_verify_s"),
                         "gov_pct_of_training": round(100 * m("t_gov_s") / m("t_fl_s"), 3),
                         "ledger_blocks": m("ledger_blocks"),
                         "ledger_kb": round(m("ledger_bytes") / 1024, 1),
                         "naive": m("naive"), "gov": m("gov"),
                         "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                         "recall": round(TP / (TP + FN), 4) if TP + FN else 0.0,
                         "honest_fpr": round(FP / (FP + TN), 4) if FP + TN else 0.0,
                         "per_seed": per})
            r = rows[-1]
            print(f"  == rounds={nr}: gov {r['t_gov_s']}s ({r['gov_pct_of_training']}% of "
                  f"training) ledger {r['ledger_kb']}KB recall {r['recall']} "
                  f"fpr {r['honest_fpr']}\n", flush=True)
    finally:
        config.NUM_ROUNDS = saved
    Path(out).write_text(json.dumps({"rounds": ROUNDS, "seeds": SEEDS, "rows": rows},
                                    indent=1), encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
