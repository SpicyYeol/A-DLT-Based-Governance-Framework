# -*- coding: utf-8 -*-
"""Cross-dataset generalization, per-subcategory capability and the
three-tier hierarchy experiment.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _agg, _conf, _flagged_set, run_seed


def run_cross_dataset(datasets: list[str] | None = None,
                      seeds: list[int] | None = None) -> dict:
    datasets = datasets or ["mnist", "fashion_mnist"]
    seeds = seeds or config.exp_seeds()
    out = []
    for name in datasets:
        per = [run_seed(s, dataset=name, ablations=False) for s in seeds]
        e1 = {k: _agg([r[k] for r in per]) for k in
              ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
               "gating_uplift")}
        e2 = {k: _agg([r["detection"][k] for r in per]) for k in
              ("precision", "recall", "f1", "false_positive_rate")}
        out.append({"dataset": per[0]["dataset"], "E1": e1, "E2": e2})
    return {"seeds": seeds, "datasets": out}


def run_subcategory_capability(seeds: list[int] | None = None,
                               profile: dict | None = None) -> dict:
    """Per-sub-category capability check (all 36). For each control, across
    seeds: mean subscore on honest vs on adversaries, adversary-flag rate,
    honest false-positive rate, and which attacks it actually catches.
    Reveals which sub-categories are active per-agent detectors vs
    system/attestation controls."""
    from ..taxonomy import ALL_CONTROLS
    import statistics as st
    seeds = seeds or config.exp_seeds()
    profile = profile or config.ADVERSARY_PROFILE
    acc = {cp.id: {"honest": [], "adv": [], "fa": [], "fh": []} for cp in ALL_CONTROLS}
    detects = {cp.id: set() for cp in ALL_CONTROLS}
    for s in seeds:
        prep = fl.prepare(s)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
        ctx = build_ctx(naive["signals"])
        results = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        gt = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
        honest = set(ids) - gt
        for cp in ALL_CONTROLS:
            pa = results[cp.id].per_agent
            if not pa:
                continue
            hs = [pa[a] for a in honest if a in pa]
            ad = [pa[a] for a in gt if a in pa]
            if hs:
                acc[cp.id]["honest"].append(st.mean(hs))
            if ad:
                acc[cp.id]["adv"].append(st.mean(ad))
            acc[cp.id]["fa"].append(1.0 if any(pa.get(a, 1) < 0.5 for a in gt) else 0.0)
            acc[cp.id]["fh"].append(1.0 if any(pa.get(a, 1) < 0.5 for a in honest) else 0.0)
            for a in gt:
                if pa.get(a, 1) < 0.5:
                    lbl = ("revoked" if a == config.REVOKED_AGENT
                           else ctx.agents[a].get("behaviour", str(a)))
                    detects[cp.id].add(lbl)
    rows = []
    for cp in ALL_CONTROLS:
        a = acc[cp.id]
        per_agent_control = bool(a["honest"] or a["adv"])
        rows.append({
            "id": cp.id, "level": cp.level, "category": cp.category,
            "subcategory": cp.subcategory, "severity": cp.severity,
            "method": cp.method, "orient": cp.orient,
            "per_agent_detector": per_agent_control,
            "honest_subscore": _agg(a["honest"]) if a["honest"] else {"mean": 1.0, "std": 0.0},
            "adversary_subscore": _agg(a["adv"]) if a["adv"] else {"mean": None, "std": None},
            "adv_flag_rate": _agg(a["fa"]) if a["fa"] else {"mean": 0.0, "std": 0.0},
            "honest_fp_rate": _agg(a["fh"]) if a["fh"] else {"mean": 0.0, "std": 0.0},
            "detects": sorted(detects[cp.id]),
            "discriminative": bool(detects[cp.id])})
    n_disc = sum(1 for r in rows if r["discriminative"])
    return {"seeds": seeds, "n_discriminative": n_disc,
            "n_attestation_or_system": len(rows) - sum(1 for r in rows
                                                       if r["per_agent_detector"]),
            "rows": rows}


def run_hierarchy(seed: int | None = None, profile: dict | None = None) -> dict:
    """H1 — demonstrate the 3-tier hierarchy (Clause 6). Produces:
      (a) score roll-up  sub-category -> category -> level;
      (b) per-agent × level detection contribution (which tier flags whom);
      (c) per-level detection ownership (which tier catches which attack);
      (d) level ablation — disable a whole tier, show it owns distinct threats.
    """
    from ..taxonomy import by_level, by_id, LEVEL_WEIGHTS
    import statistics as st
    seed = config.EXP_BASE_SEED if seed is None else seed
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    flagged_full = _flagged_set(results, ids)
    gov_acc = fl.train_trajectory(prep, exclude=flagged_full, profile=profile)["accuracy"]
    levels = ("LARGE", "MEDIUM", "SMALL")

    # (a) score roll-up
    rollup = {}
    for lvl in levels:
        cats = {}
        for cp in by_level(lvl):
            cats.setdefault(cp.category, []).append(results[cp.id].score)
        cat_means = {c: round(st.mean(v), 3) for c, v in cats.items()}
        rollup[lvl] = {"categories": cat_means,
                       "level_score": round(st.mean(list(cat_means.values())), 3),
                       "weight": LEVEL_WEIGHTS[lvl]}

    # (b) per-agent × level subscore + verdict
    per_agent_level = {}
    for aid in ids:
        d = {}
        for lvl in levels:
            subs = [results[cp.id].per_agent[aid] for cp in by_level(lvl)
                    if aid in results[cp.id].per_agent]
            d[lvl] = round(st.mean(subs), 3) if subs else 1.0
        per_agent_level[aid] = {"behaviour": ctx.agents[aid].get("behaviour"),
                                "is_adversary": aid in gt_bad,
                                "levels": d,
                                "verdict": agent_verdict(results, aid)[0]}

    # (c) per-level detection ownership
    detection_ownership = {}
    for aid in sorted(gt_bad):
        _, failed = agent_verdict(results, aid)
        flvls = sorted({by_id(cid).level for cid in failed})
        detection_ownership[aid] = {"behaviour": ctx.agents[aid].get("behaviour"),
                                    "flagged_by_levels": flvls,
                                    "controls": list(failed.keys())}

    # (d) level ablation
    level_ablation = []
    for lvl in levels:
        disabled = frozenset(cp.id for cp in by_level(lvl))
        flagged_ab = _flagged_set(results, ids, disabled=disabled)
        newly = sorted((gt_bad & flagged_full) - flagged_ab)
        acc = (fl.train_trajectory(prep, exclude=flagged_ab, profile=profile)["accuracy"]
               if newly else gov_acc)
        level_ablation.append({
            "level": lvl, "n_controls": len(disabled),
            "newly_undetected": newly,
            "undetected_behaviours": [ctx.agents[a].get("behaviour") for a in newly],
            "accuracy_under_ablation": round(acc, 4),
            "delta_vs_governance": round(acc - gov_acc, 4)})

    return {"seed": seed, "governance_accuracy": round(gov_acc, 4),
            "foundation_note": "LARGE is the system-wide gate; MEDIUM/SMALL score per agent",
            "rollup": rollup, "per_agent_level": per_agent_level,
            "detection_ownership": detection_ownership,
            "level_ablation": level_ablation}


def _e1e2_for_prep(prep: dict, profile: dict) -> dict:
    """naive/gov/oracle accuracy + detection + honest FPR for one prepared run.
    Shared by the B3 (non-IID) and deep-backbone experiments."""
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    flagged = _flagged_set(results, ids)
    honest = set(ids) - gt_bad
    fpr = len(flagged & honest) / max(len(honest), 1)
    det = _conf(flagged, ids, gt_bad)
    gov = fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"]
    oracle = fl.train_trajectory(prep, exclude=gt_bad, profile=profile)["accuracy"]
    return {"naive": naive["accuracy"], "gov": gov, "oracle": oracle,
            "recall": det["recall"], "precision": det["precision"],
            "honest_fpr": round(fpr, 4)}
