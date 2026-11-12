# -*- coding: utf-8 -*-
"""Deep-model backbone experiment: linear against MLP.

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


def run_deep_backbone(backbones: list[str] | None = None,
                      seeds: list[int] | None = None) -> dict:
    """Repeat E1/E2 with a deep (MLP) backbone to show the governance mechanics
    hold beyond the linear model."""
    backbones = backbones or ["linear", "mlp"]
    seeds = seeds or config.exp_seeds()[:2]
    profile = config.ADVERSARY_PROFILE
    rows = []
    for bb in backbones:
        acc = {"naive": [], "gov": [], "oracle": [], "recall": [], "fpr": []}
        for s in seeds:
            prep = fl.prepare(s, backbone=bb)
            r = _e1e2_for_prep(prep, profile)
            acc["naive"].append(r["naive"]); acc["gov"].append(r["gov"])
            acc["oracle"].append(r["oracle"]); acc["recall"].append(r["recall"])
            acc["fpr"].append(r["honest_fpr"])
        rows.append({"backbone": bb,
                     "naive_accuracy": _agg(acc["naive"]),
                     "governance_accuracy": _agg(acc["gov"]),
                     "oracle_accuracy": _agg(acc["oracle"]),
                     "detection_recall": _agg(acc["recall"]),
                     "honest_false_positive_rate": _agg(acc["fpr"])})
    return {"seeds": seeds, "rows": rows}
