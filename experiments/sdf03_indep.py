# -*- coding: utf-8 -*-
"""S-DF-03 gradient channel: independent-seed validation.

The threshold 0.45 was chosen by looking at seeds 42-46, so those seeds
cannot evaluate it. Here the threshold is FROZEN at 0.45 and applied to
seeds 47-71, which were never inspected. No re-tuning is permitted after
seeing these results.

Reports, for the calibration set and the held-out set separately:
  label-flipping recall, honest FPR, precision,
  estimated-KL distribution for flipper / other adversaries / honest,
  the separation gap, and the per-seed detection outcome.

Usage: python sdf03_indep.py <out.json>
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
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402

CALIB = list(range(42, 47))          # used to pick the threshold
HOLDOUT = list(range(47, 72))        # never inspected before this run
TAU = 0.45                           # FROZEN
EPS = 1e-9
FLIPPER = 1


def kl(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return float(np.sum(np.where(p > 0, p * np.log((p + EPS) / (q + EPS)), 0.0)))


def estimate_dist(delta_b, prior):
    d = np.asarray(delta_b, float)
    d = d - d.mean()
    if np.allclose(d, 0):
        return np.asarray(prior, float)
    neg = d < 0
    kappa = np.inf
    if neg.any():
        kappa = float(np.min(-np.asarray(prior)[neg] / d[neg]))
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 1.0
    est = np.clip(np.asarray(prior, float) + kappa * d, 0.0, None)
    tot = est.sum()
    return est / tot if tot > 0 else np.asarray(prior, float)


def run(seeds, tag):
    profile = config.ADVERSARY_PROFILE
    truth = set(profile) | {config.REVOKED_AGENT}
    rows = []
    for s in seeds:
        prep = fl.prepare(s)
        tr = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                 profile=profile, return_updates=True)
        ctx = build_ctx(tr["signals"])
        res = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        ncls = prep["n_classes"]
        prior = np.asarray(prep["base_dist"], float)

        est = {}
        for i in ids:
            upd = tr["updates"].get(i)
            est[i] = 0.0 if upd is None else kl(
                estimate_dist(np.asarray(upd, float)[-ncls:], prior), prior)

        # S-DF-03 replaced by the estimate; every other control unchanged
        extra = {i for i in ids if est[i] > TAU}
        flagged = set()
        for a in ids:
            v = agent_verdict(res, a, disabled=frozenset({"S-DF-03"}))[0]
            if v != "Trusted" or a in extra:
                flagged.add(a)
        tp = len(flagged & truth); fp = len(flagged - truth)
        fn = len(truth - flagged); tn = len(set(ids) - truth - flagged)
        rows.append({
            "seed": s,
            "est_kl": {str(i): round(est[i], 4) for i in ids},
            "flipper_est_kl": round(est[FLIPPER], 4),
            "flipper_caught": est[FLIPPER] > TAU,
            "honest_over_tau": sorted(i for i in ids
                                      if i not in truth and est[i] > TAU),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1) if (tp + fp) else 1.0,
            "honest_fpr": fp / max(fp + tn, 1),
            "missed": sorted(truth - flagged)})
        print("  [%s] seed %d  flipper_kl=%.4f caught=%s honest_over=%s"
              % (tag, s, est[FLIPPER], est[FLIPPER] > TAU,
                 rows[-1]["honest_over_tau"]), flush=True)

    flip = [r["flipper_est_kl"] for r in rows]
    hon, oth = [], []
    for r in rows:
        for k, v in r["est_kl"].items():
            i = int(k)
            if i == FLIPPER:
                continue
            (oth if i in truth else hon).append(v)
    gap = min(flip) - max(hon + oth) if flip and (hon + oth) else float("nan")
    return {
        "seeds": seeds, "tau": TAU, "n_seeds": len(rows),
        "flipper_recall": round(st.mean(1.0 if r["flipper_caught"] else 0.0
                                        for r in rows), 4),
        "overall_recall": round(st.mean(r["recall"] for r in rows), 4),
        "precision": round(st.mean(r["precision"] for r in rows), 4),
        "honest_fpr": round(st.mean(r["honest_fpr"] for r in rows), 4),
        "honest_fp_total": sum(r["FP"] for r in rows),
        "honest_decisions": sum(r["FP"] + r["TN"] for r in rows),
        "kl_flipper": {"min": round(min(flip), 4), "mean": round(st.mean(flip), 4),
                       "max": round(max(flip), 4)},
        "kl_honest": {"min": round(min(hon), 4), "mean": round(st.mean(hon), 4),
                      "max": round(max(hon), 4)},
        "kl_other_adv": {"min": round(min(oth), 4), "mean": round(st.mean(oth), 4),
                         "max": round(max(oth), 4)},
        "separation_gap": round(gap, 4),
        "per_seed": rows}


def main(out_path):
    print("calibration set (threshold was chosen here):")
    cal = run(CALIB, "calib")
    print("\nheld-out set (threshold frozen at %.2f):" % TAU)
    hold = run(HOLDOUT, "holdout")

    print()
    hdr = "%-12s %7s %9s %9s %9s %11s %9s"
    print(hdr % ("set", "seeds", "flip-rec", "recall", "precision",
                 "honest FPR", "gap"))
    for name, d in (("calibration", cal), ("held-out", hold)):
        print("%-12s %7d %9.3f %9.3f %9.3f %11.3f %9.4f"
              % (name, d["n_seeds"], d["flipper_recall"], d["overall_recall"],
                 d["precision"], d["honest_fpr"], d["separation_gap"]))
    print()
    for name, d in (("calibration", cal), ("held-out", hold)):
        print("%-12s flipper KL %.3f-%.3f | honest KL %.3f-%.3f | other adv %.3f-%.3f"
              % (name, d["kl_flipper"]["min"], d["kl_flipper"]["max"],
                 d["kl_honest"]["min"], d["kl_honest"]["max"],
                 d["kl_other_adv"]["min"], d["kl_other_adv"]["max"]))
        print("%-12s honest false positives: %d of %d decisions"
              % ("", d["honest_fp_total"], d["honest_decisions"]))

    Path(out_path).write_text(json.dumps(
        {"threshold": TAU, "calibration": cal, "holdout": hold},
        indent=1), encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
