# -*- coding: utf-8 -*-
"""Shared pieces: the ablation table, the detection metrics and the
single-seed and multi-seed runs every other experiment builds on.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger


ABLATIONS = [
    {"key": "no_governance", "disabled": set(), "aggregate_all": True,
     "name": "No trust governance (aggregate all agents)",
     "risk": "Degraded model accuracy + security", "primary": "accuracy"},
    {"key": "no_model_poison_defense", "disabled": {"M-BC-01", "M-SC-02"},
     "name": "No model-poisoning detection (M-BC-01 anomaly, M-SC-02 over-weight)",
     "risk": "Degraded model accuracy (gradient/scale poisoning)",
     "primary": "accuracy"},
    {"key": "no_data_poison_defense", "disabled": {"S-DF-03"},
     "name": "No data-quality drift check (S-DF-03, ISO/IEC 5259)",
     "risk": "Degraded accuracy (label/data poisoning)", "primary": "accuracy"},
    {"key": "no_freshness", "disabled": {"S-DF-02"},
     "name": "No data-freshness SLA (S-DF-02)",
     "risk": "Stale / replayed data accepted", "primary": "undetected"},
    {"key": "no_message_integrity", "disabled": {"S-MI-02", "M-CV-03"},
     "name": "No payload-hash + signature verification (S-MI-02, M-CV-03)",
     "risk": "Tampered / forged submissions undetected", "primary": "undetected"},
    {"key": "no_identity_lifecycle", "disabled": {"L-IA-03", "M-CV-02"},
     "name": "No identity-lifecycle / revocation (L-IA-03, M-CV-02)",
     "risk": "Revoked agents keep contributing", "primary": "undetected"},
]


def _flagged_set(results, agent_ids, disabled=frozenset()) -> set[int]:
    """Agents the framework would quarantine (verdict != Trusted)."""
    return {aid for aid in agent_ids
            if agent_verdict(results, aid, disabled=disabled)[0] != "Trusted"}


def _untrusted_only(results, agent_ids, disabled=frozenset()) -> set[int]:
    return {aid for aid in agent_ids
            if agent_verdict(results, aid, disabled=disabled)[0] == "Untrusted"}


def _conf(flagged: set[int], agent_ids, ground_truth_bad: set[int]) -> dict:
    good = set(agent_ids) - ground_truth_bad
    tp = len(flagged & ground_truth_bad)
    fp = len(flagged & good)
    fn = len(ground_truth_bad - flagged)
    tn = len(good - flagged)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_positive_rate": round(fpr, 4)}


def detection_metrics(results, agent_ids, ground_truth_bad: set[int]) -> dict:
    return _conf(_flagged_set(results, agent_ids), agent_ids, ground_truth_bad)


def run_seed(seed: int, profile: dict | None = None,
             dataset: str | None = None, ablations: bool = True) -> dict:
    """Run E1–E3 for a single seed. Returns a JSON-able result dict.

    ``profile`` overrides the adversary profile (C6); ``dataset`` selects the
    image dataset (C8); ``ablations=False`` skips the E3 retrains (C8)."""
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed, dataset=dataset)

    # reference (naive) pass: trains all-agent model + records ledger & signals
    naive = fl.train_trajectory(prep, exclude=frozenset(), record=True,
                                reset_ledger=True, profile=profile)
    signals = naive["signals"]

    ctx = build_ctx(signals)               # reads the ledger just written
    results = evaluate_controls(ctx)       # cid -> ControlResult
    ids = ctx.agent_ids()
    gt_bad = set(profile) | {config.REVOKED_AGENT}

    flagged = _flagged_set(results, ids)
    # ---- E1 closed-loop gating ----
    naive_acc = naive["accuracy"]
    gov_acc = fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"]
    oracle_acc = fl.train_trajectory(prep, exclude=gt_bad, profile=profile)["accuracy"]

    # ---- E2 detection metrics ----
    det = detection_metrics(results, ids, gt_bad)

    # per-agent verdict snapshot (for reference tables)
    verdicts = {}
    for aid in ids:
        v, failed = agent_verdict(results, aid)
        verdicts[aid] = {"behaviour": ctx.agents[aid].get("behaviour"),
                         "is_adversary": aid in gt_bad, "verdict": v,
                         "failed_controls": failed}

    # ---- E3 ablations ----
    ablation_rows = []
    for ab in (ABLATIONS if ablations else []):
        disabled = frozenset(ab["disabled"])
        if ab.get("aggregate_all"):
            abl_flagged = set()               # nothing quarantined
        else:
            abl_flagged = _flagged_set(results, ids, disabled=disabled)
        acc = naive_acc if ab.get("aggregate_all") else \
            fl.train_trajectory(prep, exclude=abl_flagged, profile=profile)["accuracy"]
        # bad agents that escape detection entirely under this ablation
        undetected = sorted(gt_bad - _flagged_set(results, ids, disabled=disabled)) \
            if not ab.get("aggregate_all") else sorted(gt_bad)
        ablation_rows.append({
            "key": ab["key"], "name": ab["name"], "risk": ab["risk"],
            "primary_harm": ab["primary"],
            "accuracy_under_ablation": round(acc, 4),
            "delta_vs_governance": round(acc - gov_acc, 4),
            "n_undetected_adversaries": len(undetected),
            "undetected_agents": undetected,
        })

    return {
        "seed": seed, "dataset": signals["dataset"],
        "naive_accuracy": round(naive_acc, 4),
        "governance_accuracy": round(gov_acc, 4),
        "oracle_accuracy": round(oracle_acc, 4),
        "gating_uplift": round(gov_acc - naive_acc, 4),
        "flagged_agents": sorted(flagged),
        "detection": det,
        "verdicts": verdicts,
        "ablations": ablation_rows,
    }


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    return {"mean": round(stats.mean(values), 4),
            "std": round(stats.pstdev(values), 4) if len(values) > 1 else 0.0,
            "n": len(values)}


def run_all(seeds: list[int] | None = None) -> dict:
    seeds = seeds or config.exp_seeds()
    per_seed = [run_seed(s) for s in seeds]

    # aggregate E1
    e1 = {k: _agg([r[k] for r in per_seed]) for k in
          ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
           "gating_uplift")}
    # aggregate E2
    e2 = {k: _agg([r["detection"][k] for r in per_seed]) for k in
          ("precision", "recall", "f1", "false_positive_rate")}
    # aggregate E3 by ablation key
    e3 = {}
    for ab in ABLATIONS:
        rows = [next(x for x in r["ablations"] if x["key"] == ab["key"])
                for r in per_seed]
        e3[ab["key"]] = {
            "name": ab["name"], "risk": ab["risk"],
            "primary_harm": ab["primary"],
            "accuracy_under_ablation": _agg([x["accuracy_under_ablation"] for x in rows]),
            "delta_vs_governance": _agg([x["delta_vs_governance"] for x in rows]),
            "n_undetected_adversaries": _agg([x["n_undetected_adversaries"] for x in rows]),
        }
    return {"seeds": seeds, "per_seed": per_seed,
            "summary": {"E1_closed_loop": e1, "E2_detection": e2,
                        "E3_ablation": e3}}
