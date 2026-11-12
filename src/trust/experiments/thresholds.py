# -*- coding: utf-8 -*-
"""Threshold sensitivity for the update-anomaly control.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _conf, _flagged_set


def _anomaly_z(signals: dict) -> dict:
    """Robust modified z-score of each agent's mean update norm (same statistic
    M-BC-01 uses), computed straight from recorded signals."""
    import numpy as np
    agents = {int(k): v for k, v in signals["agents"].items()}
    ids = sorted(agents)
    norms = np.array([np.mean([r["update_norm"] for r in agents[i]["rounds"]])
                      for i in ids])
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) or 1e-9
    z = 0.6745 * (norms - med) / mad
    return {i: float(abs(zi)) for i, zi in zip(ids, z)}


def run_threshold_sweep(seed: int, taus: list[float] | None = None,
                        profile: dict | None = None) -> dict:
    """Sweep the anomaly z-threshold tau under the stealth profile and report
    detection ROC — both for the anomaly control alone and for the full
    framework (anomaly retuned + all other controls)."""
    profile = config.STEALTH_PROFILE if profile is None else profile
    taus = taus or [4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5]
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    zmap = _anomaly_z(signals)
    stealth_ids = {a for a, b in profile.items() if b == "stealth_scale"}

    rows = []
    for tau in taus:
        # anomaly detector alone: flag if |z| > tau
        anomaly_flagged = {i for i in ids if zmap[i] > tau}
        ano = _conf(anomaly_flagged, ids, gt_bad)
        # full framework: override M-BC-01 per-agent at this tau, re-verdict
        base = {i: results["M-BC-01"].per_agent.get(i, 1.0) for i in ids}
        for i in ids:
            results["M-BC-01"].per_agent[i] = 0.0 if zmap[i] > tau else 1.0
        fw_flagged = _flagged_set(results, ids)
        fw = _conf(fw_flagged, ids, gt_bad)
        for i in ids:                       # restore
            results["M-BC-01"].per_agent[i] = base[i]
        rows.append({
            "tau": tau,
            "anomaly_recall": ano["recall"], "anomaly_fpr": ano["false_positive_rate"],
            "framework_recall": fw["recall"], "framework_precision": fw["precision"],
            "framework_fpr": fw["false_positive_rate"], "framework_f1": fw["f1"],
            "stealth_detected": bool(stealth_ids <= anomaly_flagged),
        })
    # recommend the most conservative (largest) tau that maximises framework F1
    # while keeping FPR == 0 -> closes the stealth gap with fewest interventions
    zero_fp = [r for r in rows if r["framework_fpr"] == 0.0]
    pick = max(zero_fp or rows, key=lambda r: (r["framework_f1"], r["tau"]))
    return {"seed": seed, "z_scores": {str(k): round(v, 2) for k, v in zmap.items()},
            "stealth_agents": sorted(stealth_ids), "rows": rows,
            "recommended_tau": pick["tau"], "recommended": pick}
