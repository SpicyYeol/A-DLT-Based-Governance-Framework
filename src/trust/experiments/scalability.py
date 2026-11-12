# -*- coding: utf-8 -*-
"""Scalability and overhead against the number of participants.

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


def _measure_scale(seed: int) -> dict:
    import time
    from ..ledger import Ledger
    profile = config.ADVERSARY_PROFILE
    t0 = time.perf_counter()
    prep = fl.prepare(seed)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    t_fl = time.perf_counter() - t0
    signals = naive["signals"]

    t0 = time.perf_counter()
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    flagged = _flagged_set(results, ids)
    t_gov = time.perf_counter() - t0

    ledger = Ledger()
    t0 = time.perf_counter()
    chain = ledger.verify_chain()
    t_verify = time.perf_counter() - t0
    ledger_bytes = ledger.path.stat().st_size if ledger.path.exists() else 0

    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    det = _conf(flagged, ids, gt_bad)
    return {"t_prepare_s": round(t_prep, 3), "t_fl_train_s": round(t_fl, 3),
            "t_governance_s": round(t_gov, 4), "t_chain_verify_s": round(t_verify, 4),
            "ledger_blocks": chain["blocks"], "ledger_bytes": ledger_bytes,
            "detection_recall": det["recall"]}


def run_scalability(agent_counts: list[int] | None = None,
                    round_counts: list[int] | None = None,
                    seed: int | None = None) -> dict:
    seed = config.EXP_BASE_SEED if seed is None else seed
    agent_counts = agent_counts or [4, 8, 16, 32]
    round_counts = round_counts or [1, 2, 4, 8]
    saved_a, saved_r = config.NUM_AGENTS, config.NUM_ROUNDS
    rows = []
    try:
        for na in agent_counts:                 # sweep #agents (rounds fixed)
            config.NUM_AGENTS, config.NUM_ROUNDS = na, saved_r
            m = _measure_scale(seed)
            rows.append({"sweep": "agents", "num_agents": na,
                         "num_rounds": saved_r, **m})
        for nr in round_counts:                 # sweep #rounds (agents fixed)
            config.NUM_AGENTS, config.NUM_ROUNDS = saved_a, nr
            m = _measure_scale(seed)
            rows.append({"sweep": "rounds", "num_agents": saved_a,
                         "num_rounds": nr, **m})
    finally:
        config.NUM_AGENTS, config.NUM_ROUNDS = saved_a, saved_r
    return {"seed": seed, "rows": rows}
