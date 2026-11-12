# -*- coding: utf-8 -*-
"""EXP-4  taxonomy justification (weight sensitivity + reduced control set)
         -> R4-3, R1-2, R3-2
   EXP-5  seed-level confidence intervals -> R4-5

No retraining of new conditions beyond one control-evaluation pass per seed.
Usage:  python exp45.py <out.json>
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

from trust import config, taxonomy                      # noqa: E402
from trust import fl_simulation as fl                   # noqa: E402
from trust.engine import build_ctx, evaluate_controls, agent_verdict, score  # noqa: E402
from trust.taxonomy import ALL_CONTROLS                 # noqa: E402

ALL_IDS = [cp.id for cp in ALL_CONTROLS]
SEEDS = [42, 43, 44, 45, 46]

WEIGHT_SETS = {
    "default_040_035_025": {"LARGE": 0.40, "MEDIUM": 0.35, "SMALL": 0.25},
    "uniform_033": {"LARGE": 1 / 3, "MEDIUM": 1 / 3, "SMALL": 1 / 3},
    "granular_heavy": {"LARGE": 0.25, "MEDIUM": 0.35, "SMALL": 0.40},
    "structural_heavy": {"LARGE": 0.50, "MEDIUM": 0.30, "SMALL": 0.20},
}


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n <= 2:
        return 1.0
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def boot_ci(vals, B=10000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(st.mean(rng.choices(vals, k=n)) for _ in range(B))
    return [round(means[int(0.025 * B)], 4), round(means[int(0.975 * B)], 4)]


def build_cache():
    cache = []
    for s in SEEDS:
        prep = fl.prepare(s)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                    profile=config.ADVERSARY_PROFILE)
        ctx = build_ctx(naive["signals"])
        res = evaluate_controls(ctx)
        cache.append((s, ctx, res))
        print("  seed %d control evaluation done" % s, flush=True)
    return cache


def exp4a(cache):
    base = None
    rows = []
    for wname, wmap in WEIGHT_SETS.items():
        taxonomy.LEVEL_WEIGHTS.clear()
        taxonomy.LEVEL_WEIGHTS.update(wmap)
        per = []
        for s, ctx, res in cache:
            sc = score(ctx, res)
            reps = [sc["reputation"][a] for a in sorted(sc["reputation"])]
            verd = {str(a): sc["agents"][a]["verdict"] for a in sc["agents"]}
            per.append({"seed": s, "foundation": sc["foundation_score"],
                        "reputation": reps, "verdicts": verd})
        flat = [r for p in per for r in p["reputation"]]
        row = {"weights": wname, "weight_map": wmap,
               "foundation_mean": round(st.mean([p["foundation"] for p in per]), 4),
               "reputation_mean": round(st.mean(flat), 4)}
        if base is None:
            base = per
            row["verdict_changes_vs_default"] = 0
            row["rank_spearman_vs_default"] = 1.0
        else:
            ch = 0
            for i, p in enumerate(per):
                for a, v in p["verdicts"].items():
                    if v != base[i]["verdicts"][a]:
                        ch += 1
            rho = st.mean([spearman(p["reputation"], base[i]["reputation"])
                           for i, p in enumerate(per)])
            row["verdict_changes_vs_default"] = ch
            row["rank_spearman_vs_default"] = round(rho, 4)
        row["per_seed"] = per
        rows.append(row)
        print("  [weights] %-22s foundation=%.4f rep=%.4f verdict_changes=%d spearman=%s"
              % (wname, row["foundation_mean"], row["reputation_mean"],
                 row["verdict_changes_vs_default"], row["rank_spearman_vs_default"]),
              flush=True)
    taxonomy.LEVEL_WEIGHTS.clear()
    taxonomy.LEVEL_WEIGHTS.update(WEIGHT_SETS["default_040_035_025"])
    return rows


def exp4b(cache):
    fired = set()
    for s, ctx, res in cache:
        for cid in ALL_IDS:
            pa = res[cid].per_agent
            if any(v < 0.5 for v in pa.values()):
                fired.add(cid)
    never = [c for c in ALL_IDS if c not in fired]
    disabled = frozenset(never)
    mism = 0
    decisions = 0
    for s, ctx, res in cache:
        for a in ctx.agent_ids():
            decisions += 1
            if agent_verdict(res, a)[0] != agent_verdict(res, a, disabled=disabled)[0]:
                mism += 1
    out = {"n_controls_total": len(ALL_IDS), "n_fired": len(fired),
           "n_never_fired": len(never), "fired_controls": sorted(fired),
           "never_fired_controls": never,
           "verdict_mismatches_when_disabled": mism,
           "n_agent_decisions": decisions}
    print("\n  [reduced] fired %d/%d, disabling the %d never-fired controls "
          "changed %d of %d verdicts"
          % (len(fired), len(ALL_IDS), len(never), mism, decisions), flush=True)
    return out


def exp5():
    p = TB / "data_rev2" / "reports_rev1" / "rev1_seeds.json"
    if not p.exists():
        print("  [CI] missing %s - run Phase A first" % p, flush=True)
        return {"error": "missing %s" % p}
    rows = json.load(open(p, encoding="utf-8"))["rows"]
    TP = sum(r["TP"] for r in rows)
    FN = sum(r["FN"] for r in rows)
    FP = sum(r["FP"] for r in rows)
    TN = sum(r["TN"] for r in rows)
    rec_seed = [r["TP"] / (r["TP"] + r["FN"]) if (r["TP"] + r["FN"]) else 0.0
                for r in rows]
    fpr_seed = [r["FP"] / (r["FP"] + r["TN"]) if (r["FP"] + r["TN"]) else 0.0
                for r in rows]
    out = {
        "n_seeds": len(rows),
        "pooled": {
            "recall": {"k": TP, "n": TP + FN, "point": round(TP / (TP + FN), 4),
                       "clopper_pearson_95": clopper_pearson(TP, TP + FN)},
            "honest_fpr": {"k": FP, "n": FP + TN,
                           "point": round(FP / (FP + TN), 4),
                           "clopper_pearson_95": clopper_pearson(FP, FP + TN)}},
        "seed_level": {
            "recall": {"mean": round(st.mean(rec_seed), 4),
                       "bootstrap_95": boot_ci(rec_seed)},
            "honest_fpr": {"mean": round(st.mean(fpr_seed), 4),
                           "bootstrap_95": boot_ci(fpr_seed)}},
    }
    print("\n  [CI] recall pooled CP %s  vs seed-level bootstrap %s"
          % (out["pooled"]["recall"]["clopper_pearson_95"],
             out["seed_level"]["recall"]["bootstrap_95"]), flush=True)
    print("  [CI] fpr    pooled CP %s  vs seed-level bootstrap %s"
          % (out["pooled"]["honest_fpr"]["clopper_pearson_95"],
             out["seed_level"]["honest_fpr"]["bootstrap_95"]), flush=True)
    return out


def main(out_path):
    cache = build_cache()
    rep = {"seeds": SEEDS,
           "exp4a_weight_sensitivity": exp4a(cache),
           "exp4b_reduced_set": exp4b(cache),
           "exp5_intervals": exp5()}
    Path(out_path).write_text(json.dumps(rep, indent=1, default=str),
                              encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
