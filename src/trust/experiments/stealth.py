# -*- coding: utf-8 -*-
"""Stealth and collusion attack profile.

Split out of the single experiments module; the code is unchanged.
"""
from __future__ import annotations

import statistics as stats

from .. import config
from .. import fl_simulation as fl
from .. import aggregators as agg
from ..engine import build_ctx, evaluate_controls, agent_verdict
from ..ledger import Ledger
from ._core import _agg, run_seed


def run_stealth(seeds: list[int] | None = None) -> dict:
    """Run E1/E2 under the harder STEALTH_PROFILE (low-magnitude scale +
    colluding label-flip) to probe detection blind spots."""
    seeds = seeds or config.exp_seeds()
    profile = config.STEALTH_PROFILE
    per_seed = [run_seed(s, profile=profile) for s in seeds]
    e1 = {k: _agg([r[k] for r in per_seed]) for k in
          ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
           "gating_uplift")}
    e2 = {k: _agg([r["detection"][k] for r in per_seed]) for k in
          ("precision", "recall", "f1", "false_positive_rate")}
    # per-behaviour detection: was each adversary flagged?
    behaviour_detection = {}
    for r in per_seed:
        for aid, v in r["verdicts"].items():
            if v["is_adversary"]:
                beh = v["behaviour"]
                behaviour_detection.setdefault(beh, []).append(
                    1 if v["verdict"] != "Trusted" else 0)
    beh_rate = {b: round(sum(x) / len(x), 3) for b, x in behaviour_detection.items()}
    beh_std = {b: round(_agg(x)["std"], 3) for b, x in behaviour_detection.items()}
    return {"seeds": seeds, "profile": {str(k): v for k, v in profile.items()},
            "per_seed": per_seed, "E1_closed_loop": e1, "E2_detection": e2,
            "per_behaviour_detection_rate": beh_rate,
            "per_behaviour_detection_std": beh_std}
