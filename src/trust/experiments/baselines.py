# -*- coding: utf-8 -*-
"""Baseline robust-aggregation comparison.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _agg, _flagged_set, detection_metrics


def run_baselines_seed(seed: int, profile: dict | None = None) -> dict:
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | {config.REVOKED_AGENT}
    flagged = _flagged_set(results, ids)

    acc = {}
    for name, fn in agg.BASELINES.items():          # robust aggregators, all agents
        acc[name] = round(fl.train_trajectory(
            prep, profile=profile, aggregator=fn)["accuracy"], 4)
    acc["Governance (ours)"] = round(fl.train_trajectory(
        prep, exclude=flagged, profile=profile)["accuracy"], 4)
    acc["Oracle"] = round(fl.train_trajectory(
        prep, exclude=gt_bad, profile=profile)["accuracy"], 4)

    return {"seed": seed, "accuracy": acc,
            "detection_ours": detection_metrics(results, ids, gt_bad),
            "flagged": sorted(flagged), "gt_bad": sorted(gt_bad)}


def run_baselines(seeds: list[int] | None = None, profile: dict | None = None) -> dict:
    seeds = seeds or config.exp_seeds()
    per_seed = [run_baselines_seed(s, profile) for s in seeds]
    names = list(per_seed[0]["accuracy"].keys())
    summary = {n: _agg([r["accuracy"][n] for r in per_seed]) for n in names}
    det = {k: _agg([r["detection_ours"][k] for r in per_seed]) for k in
           ("precision", "recall", "f1", "false_positive_rate")}
    return {"seeds": seeds, "per_seed": per_seed,
            "summary_accuracy": summary, "detection_ours": det}
