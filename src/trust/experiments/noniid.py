# -*- coding: utf-8 -*-
"""Non-IID severity sweep over the Dirichlet concentration.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _agg


def run_noniid(alphas: list[float] | None = None,
               seeds: list[int] | None = None) -> dict:
    """Sweep data heterogeneity. Small alpha = extreme non-IID. Watches whether
    honest-but-skewed agents get false-flagged (the global-prior drift control
    S-DF-03 assumes IID)."""
    alphas = alphas or [10.0, 1.0, 0.5, 0.1]
    seeds = seeds or config.exp_seeds()[:2]
    profile = config.ADVERSARY_PROFILE
    rows = []
    for a in alphas:
        acc = {"naive": [], "gov": [], "oracle": [], "recall": [], "fpr": []}
        for s in seeds:
            prep = fl.prepare(s, partition="dirichlet", alpha=a)
            r = _e1e2_for_prep(prep, profile)
            acc["naive"].append(r["naive"]); acc["gov"].append(r["gov"])
            acc["oracle"].append(r["oracle"]); acc["recall"].append(r["recall"])
            acc["fpr"].append(r["honest_fpr"])
        rows.append({"alpha": a,
                     "naive_accuracy": _agg(acc["naive"]),
                     "governance_accuracy": _agg(acc["gov"]),
                     "oracle_accuracy": _agg(acc["oracle"]),
                     "adversary_recall": _agg(acc["recall"]),
                     "honest_false_positive_rate": _agg(acc["fpr"])})
    return {"seeds": seeds, "rows": rows}


def run_noniid_fix(alphas: list[float] | None = None,
                   seeds: list[int] | None = None) -> dict:
    """B3 remedy: compare the label-prior drift control (current S-DF-03) with a
    non-IID-aware FLTrust server-anchored direction control across the α sweep.
    Reports honest false-positive rate and directional-poison recall for each."""
    import numpy as np
    from .. import fltrust
    alphas = alphas or [10.0, 1.0, 0.5, 0.1]
    seeds = seeds or config.exp_seeds()[:2]
    profile = config.ADVERSARY_PROFILE
    directional = {a for a, b in profile.items()
                   if b in ("label_flip", "collude_flip")}
    rows = []
    for al in alphas:
        agg = {"fpr_lp": [], "rec_lp": [], "fpr_ft": [], "rec_ft": []}
        for s in seeds:
            prep = fl.prepare(s, partition="dirichlet", alpha=al)
            naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                        profile=profile, return_updates=True,
                                        return_server_update=True)
            ctx = build_ctx(naive["signals"])
            ids = ctx.agent_ids()
            honest = set(ids) - set(profile) - {config.REVOKED_AGENT}
            # (a) label-prior detector = current S-DF-03 rule (mean label KL > 0.2)
            kl = {i: float(np.mean([r["label_kl"] for r in ctx.agents[i]["rounds"]]))
                  for i in ids}
            lp = {i for i in ids if kl[i] > 0.2}
            # (b) FLTrust server-anchored direction detector
            fts = fltrust.fltrust_scores(naive["updates"], naive["server_update"])
            ft = fltrust.fltrust_directional_flagged(fts, tau=0.0)

            def fpr(f):
                return len(f & honest) / max(len(honest), 1)

            def rec(f):
                return len(f & directional) / max(len(directional), 1)

            agg["fpr_lp"].append(fpr(lp)); agg["rec_lp"].append(rec(lp))
            agg["fpr_ft"].append(fpr(ft)); agg["rec_ft"].append(rec(ft))
        rows.append({"alpha": al,
                     "labelprior_honest_fpr": _agg(agg["fpr_lp"]),
                     "labelprior_recall": _agg(agg["rec_lp"]),
                     "fltrust_honest_fpr": _agg(agg["fpr_ft"]),
                     "fltrust_recall": _agg(agg["rec_ft"])})
    return {"seeds": seeds, "rows": rows}
