# -*- coding: utf-8 -*-
"""Conventional-checks baseline, and the split between control families.

First part: compare the framework against a competitor that runs the
conventional security checks and robust aggregation *without* the governance
framework. The comparison reuses the framework's own evaluated path (the
`disabled` argument of `_flagged_set`), so the only difference between the
conditions is which control points are consulted.

Second part: separate the share of the overall recall that comes from the
deterministic controls (cryptographic and registry predicates) from the share
that comes from the statistical controls (thresholds and distributions). This
is the decomposition reported as the control-family table (Table 15 of the
paper).

Writes one JSON file.  Usage:  python exp12.py <out.json>
"""
from __future__ import annotations
import os, sys, json, statistics as st
from pathlib import Path

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp12"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config
from trust import fl_simulation as fl
from trust import aggregators as agg
from trust.engine import build_ctx, evaluate_controls
from trust.taxonomy import ALL_CONTROLS
from trust.experiments import _flagged_set, _conf

ALL_IDS = {cp.id for cp in ALL_CONTROLS}

# ---- the conventional checks a deployed FL pipeline usually runs ---------
#   authentication-tag verification (an HMAC tag, not a digital signature) /
#   payload hash / freshness bound / credential revocation / identity lifecycle
STANDARD = {"M-CV-03", "S-MI-02", "S-DF-02", "M-CV-02", "L-IA-03"}

# ---- classification of the controls by the kind of judgment they make ----
STATISTICAL = {"M-BC-01", "M-BC-02", "M-SC-02", "S-DF-03"}
ATTESTATION = {cp.id for cp in ALL_CONTROLS if cp.method == "attestation"}
DETERMINISTIC = ALL_IDS - STATISTICAL - ATTESTATION

assert STANDARD <= DETERMINISTIC, STANDARD - DETERMINISTIC
assert not (STATISTICAL & ATTESTATION)

def off(keep: set) -> frozenset:
    return frozenset(ALL_IDS - keep)

AGGS = {"FedAvg": agg.fedavg, "Multi-Krum": agg.multi_krum,
        "Trimmed-Mean": agg.trimmed_mean}

PROFILES = {
    "P1_primary":   dict(profile=config.ADVERSARY_PROFILE, revoked=7, prep={}),
    "P2_stealth":   dict(profile=config.STEALTH_PROFILE,   revoked=7, prep={}),
    "P3_noniid":    dict(profile=config.ADVERSARY_PROFILE, revoked=7,
                         prep=dict(partition="dirichlet", alpha=0.5)),
    "P4_collusion": dict(profile={1: "scale_attack", 3: "scale_attack",
                                  5: "scale_attack", 2: "scale_attack",
                                  6: "tamper"},
                         revoked=-1, prep={}),
}
SEEDS = [42, 43, 44, 45, 46]


def one(seed: int, spec: dict) -> dict:
    config.REVOKED_AGENT = spec["revoked"]
    profile = spec["profile"]
    prep = fl.prepare(seed, **spec["prep"])
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile)
    ctx = build_ctx(naive["signals"])
    res = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt = set(profile) | ({spec["revoked"]} if spec["revoked"] >= 0 else set())

    sets = {
        "governance_full": _flagged_set(res, ids),
        "standard_only":   _flagged_set(res, ids, disabled=off(STANDARD)),
        "deterministic":   _flagged_set(res, ids, disabled=off(DETERMINISTIC)),
        "statistical":     _flagged_set(res, ids, disabled=off(STATISTICAL)),
    }

    def acc(exclude, aggregator=None):
        return round(fl.train_trajectory(prep, exclude=exclude, profile=profile,
                                         aggregator=aggregator)["accuracy"], 4)

    accs = {"naive": round(naive["accuracy"], 4),
            "oracle": acc(gt),
            "governance+FedAvg": acc(sets["governance_full"])}
    for an, af in AGGS.items():
        accs[f"robust_alone/{an}"] = acc(frozenset(), af)          # no checks at all
        accs[f"standard+{an}"] = acc(sets["standard_only"], af)    # competing baseline

    # per-agent attribution: which control flagged the participant
    attribution = {}
    for aid in sorted(gt):
        fired = sorted(cid for cid in ALL_IDS
                       if aid in res[cid].per_agent
                       and res[cid].per_agent[aid] < 0.5)
        attribution[aid] = {
            "behaviour": ctx.agents[aid].get("behaviour"),
            "fired": fired,
            "by_standard": bool(set(fired) & STANDARD),
            "by_statistical": bool(set(fired) & STATISTICAL),
            "by_deterministic": bool(set(fired) & DETERMINISTIC),
        }

    return {"seed": seed, "accuracy": accs, "gt_bad": sorted(gt),
            "detection": {k: _conf(v, ids, gt) for k, v in sets.items()},
            "flagged": {k: sorted(v) for k, v in sets.items()},
            "attribution": attribution}


def agg_stat(vals):
    return {"mean": round(st.mean(vals), 4),
            "std": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0}


def main(out_path: str) -> int:
    report = {"config": {"image_cap": os.environ["TRUST_IMAGE_CAP"],
                         "seeds": SEEDS,
                         "standard_checks": sorted(STANDARD),
                         "statistical_controls": sorted(STATISTICAL),
                         "attestation_controls": sorted(ATTESTATION),
                         "n_deterministic": len(DETERMINISTIC)},
              "profiles": {}}
    for pname, spec in PROFILES.items():
        print(f"\n===== {pname} =====", flush=True)
        rows = []
        for s in SEEDS:
            r = one(s, spec)
            rows.append(r)
            d = r["detection"]
            print(f"  seed {s}: gov_rec={d['governance_full']['recall']:.2f} "
                  f"std_rec={d['standard_only']['recall']:.2f} "
                  f"det_rec={d['deterministic']['recall']:.2f} "
                  f"stat_rec={d['statistical']['recall']:.2f} | "
                  f"std+MK={r['accuracy']['standard+Multi-Krum']:.4f} "
                  f"gov={r['accuracy']['governance+FedAvg']:.4f}", flush=True)
        keys = rows[0]["accuracy"].keys()
        summary_acc = {k: agg_stat([r["accuracy"][k] for r in rows]) for k in keys}
        summary_det = {}
        for sk in rows[0]["detection"]:
            summary_det[sk] = {m: agg_stat([r["detection"][sk][m] for r in rows])
                               for m in ("precision", "recall", "f1",
                                         "false_positive_rate")}
        report["profiles"][pname] = {"per_seed": rows,
                                     "summary_accuracy": summary_acc,
                                     "summary_detection": summary_det}
        print("  -- summary --")
        for k in ("naive", "governance+FedAvg", "standard+Multi-Krum",
                  "robust_alone/Multi-Krum", "oracle"):
            v = summary_acc[k]
            print(f"     {k:26s} {v['mean']:.4f}+-{v['std']:.4f}")
        for sk in ("governance_full", "standard_only", "deterministic", "statistical"):
            v = summary_det[sk]["recall"]
            f = summary_det[sk]["false_positive_rate"]
            print(f"     recall[{sk:16s}] {v['mean']:.3f}  fpr {f['mean']:.3f}")

    Path(out_path).write_text(json.dumps(report, indent=1, default=str),
                              encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
