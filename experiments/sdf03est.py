# -*- coding: utf-8 -*-
"""Deployment channel 3 for S-DF-03: estimate a participant's label
distribution from its submitted update, with no access to its labels.

For a softmax / one-vs-rest output layer trained with cross-entropy, the
gradient on the output bias for class k is

    dL/db_k = mean_i (p_ik - y_ik) = (mean predicted prob)_k - (label freq)_k

so the bias component of the submitted update carries the difference between
how often a class actually appeared and how often the current model expected
it. The server therefore recovers a label-frequency signal from the update
alone.

We compare three detectors on the same rounds:
  oracle      S-DF-03 as implemented in the testbed (reads the local labels)
  estimated   the same rule applied to the update-derived estimate
  none        S-DF-03 disabled

Usage: python sdf03est.py <out.json>
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

SEEDS = [42, 43, 44, 45, 46]
EPS = 1e-9


def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.where(p > 0, p * np.log((p + EPS) / (q + EPS)), 0.0)))


def estimate_dist(delta_b, prior):
    """Recover a label distribution from the bias part of an update.

    freq_k ~ prior_k + kappa * (db_k - mean db).  kappa is the unknown
    protocol scale (learning rate x local steps); the server picks the
    smallest kappa that makes the estimate a valid distribution, which is
    the most conservative reading of the signal.
    """
    d = np.asarray(delta_b, dtype=float)
    d = d - d.mean()
    if np.allclose(d, 0):
        return np.asarray(prior, dtype=float)
    # largest kappa keeping every component non-negative
    neg = d < 0
    kappa = np.inf
    if neg.any():
        kappa = float(np.min(-np.asarray(prior)[neg] / d[neg]))
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 1.0
    est = np.asarray(prior, dtype=float) + kappa * d
    est = np.clip(est, 0.0, None)
    tot = est.sum()
    return est / tot if tot > 0 else np.asarray(prior, dtype=float)


def main(out_path):
    profile = config.ADVERSARY_PROFILE
    truth = set(profile) | {config.REVOKED_AGENT}
    rows = []
    det = {"oracle": [], "estimated": [], "none": []}

    for s in SEEDS:
        prep = fl.prepare(s)
        tr = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                 profile=profile, return_updates=True)
        ctx = build_ctx(tr["signals"])
        res = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        ncls = prep["n_classes"]
        prior = np.asarray(prep["base_dist"], dtype=float)

        # true (oracle) divergence recorded by the testbed
        true_kl = {i: float(np.mean([r["label_kl"]
                                     for r in ctx.agents[i]["rounds"]]))
                   for i in ids}
        # update-derived estimate
        est_kl = {}
        for i in ids:
            upd = tr["updates"].get(i)
            if upd is None:
                est_kl[i] = 0.0
                continue
            db = np.asarray(upd, dtype=float)[-ncls:]
            est_kl[i] = kl(estimate_dist(db, prior), prior)

        # detection with each S-DF-03 variant, all other controls unchanged
        for name in ("oracle", "estimated", "none"):
            if name == "oracle":
                extra = {i for i in ids if true_kl[i] > 0.20}
            elif name == "estimated":
                extra = {i for i in ids if est_kl[i] > 0.20}
            else:
                extra = set()
            flagged = set()
            for a in ids:
                v = agent_verdict(res, a, disabled=frozenset({"S-DF-03"}))[0]
                if v != "Trusted" or a in extra:
                    flagged.add(a)
            tp = len(flagged & truth)
            fp = len(flagged - truth)
            fn = len(truth - flagged)
            det[name].append({"seed": s, "TP": tp, "FP": fp, "FN": fn,
                              "recall": tp / max(tp + fn, 1),
                              "missed": sorted(truth - flagged)})

        rows.append({"seed": s,
                     "true_kl": {str(i): round(true_kl[i], 4) for i in ids},
                     "est_kl": {str(i): round(est_kl[i], 4) for i in ids}})
        print("  seed %d done" % s, flush=True)

    # agreement between the oracle and the estimate
    t_all, e_all = [], []
    for r in rows:
        for i in sorted(r["true_kl"], key=int):
            t_all.append(r["true_kl"][i])
            e_all.append(r["est_kl"][i])
    corr = float(np.corrcoef(t_all, e_all)[0, 1])

    print()
    print("%-28s %8s %8s" % ("agent (behaviour)", "true KL", "est KL"))
    lab = {0: "honest", 1: "label flip", 2: "honest", 3: "scaling",
           4: "honest", 5: "replay", 6: "tamper", 7: "revoked"}
    for i in range(config.NUM_AGENTS):
        t = st.mean(r["true_kl"][str(i)] for r in rows)
        e = st.mean(r["est_kl"][str(i)] for r in rows)
        print("%-28s %8.4f %8.4f" % ("%d  %s" % (i, lab.get(i, "?")), t, e))
    print("\nPearson correlation(true, estimated) over %d agent-seeds: %.3f"
          % (len(t_all), corr))

    print()
    print("%-12s %8s %8s %8s" % ("S-DF-03", "recall", "FP", "missed"))
    summary = {}
    for name in ("oracle", "estimated", "none"):
        rs = det[name]
        rec = st.mean(r["recall"] for r in rs)
        fp = sum(r["FP"] for r in rs)
        missed = sorted({m for r in rs for m in r["missed"]})
        summary[name] = {"recall_mean": round(rec, 4),
                         "recall_sd": round(st.pstdev([r["recall"] for r in rs]), 4),
                         "FP_total": fp, "missed_agents": missed}
        print("%-12s %8.4f %8d  %s" % (name, rec, fp, missed))

    Path(out_path).write_text(json.dumps(
        {"seeds": SEEDS, "correlation_true_vs_estimated": round(corr, 4),
         "detection": summary, "per_seed": rows}, indent=1), encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
