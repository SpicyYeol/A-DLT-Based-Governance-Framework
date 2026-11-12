"""Governance engine — runs all 36 controls and produces the trust verdict.

Ties Clause 7 (Trust Control Structure: monitoring / enforcement / audit) to
Clause 8 (Control Criteria: metrics / thresholds / conformance) and to the
the three control outcomes: Trusted / Under Review / Untrusted.
"""

from __future__ import annotations

import importlib
import numpy as np

from . import config
from .taxonomy import ALL_CONTROLS, LEVEL_WEIGHTS, TRUST_THRESHOLDS, by_id

FAIL_SUBSCORE = 0.5   # a per-agent subscore below this counts as a control failure
from .checks_base import Ctx, ControlResult
from .attestations import evaluate_attestation
from .ledger import Ledger

_MODULES = {
    "large": "trust.controls.large",
    "medium": "trust.controls.medium",
    "small": "trust.controls.small",
}


def _resolve(dotted: str):
    mod_key, func = dotted.split(".", 1)
    mod = importlib.import_module(_MODULES[mod_key])
    return getattr(mod, func)


def build_ctx(signals: dict) -> Ctx:
    ledger = Ledger()
    blocks = ledger.read()
    chain = ledger.verify_chain()
    tabular = signals.get("tabular", {})
    return Ctx(signals, blocks, chain, tabular, config)


def evaluate_controls(ctx: Ctx) -> dict[str, ControlResult]:
    results: dict[str, ControlResult] = {}
    for cp in ALL_CONTROLS:
        if cp.method == "attestation" or not cp.check:
            results[cp.id] = evaluate_attestation(cp.id)
        else:
            results[cp.id] = _resolve(cp.check)(ctx)
    return results


def _agent_behaviour_score(agent_id: int, results: dict[str, ControlResult]) -> dict:
    """Level-weighted mean of this agent's per-control subscores (MEDIUM+SMALL)."""
    num, den = 0.0, 0.0
    contributing = {}
    for cp in ALL_CONTROLS:
        r = results[cp.id]
        if agent_id in r.per_agent:
            w = LEVEL_WEIGHTS[cp.level] * cp.weight
            s = r.per_agent[agent_id]
            num += w * s
            den += w
            if s < 0.80:   # only surface material findings
                contributing[cp.id] = round(s, 3)
    score = (num / den) if den else 1.0
    return {"score": score, "weak_controls": contributing}


def agent_verdict(results: dict[str, ControlResult], aid: int, *,
                  gate_ok: bool = True,
                  disabled: frozenset = frozenset()) -> tuple[str, dict]:
    """Severity-driven verdict for one agent. ``disabled`` control ids are
    ignored — used by ablation experiments to simulate a missing control."""
    failed = [cp for cp in ALL_CONTROLS
              if cp.id not in disabled
              and aid in results[cp.id].per_agent
              and results[cp.id].per_agent[aid] < FAIL_SUBSCORE]
    severities = {cp.severity for cp in failed}
    if not gate_ok or "reject" in severities:
        verdict = "Untrusted"
    elif "review" in severities:
        verdict = "Under Review"
    else:
        verdict = "Trusted"
    return verdict, {cp.id: cp.severity for cp in failed}


def score(ctx: Ctx, results: dict[str, ControlResult]) -> dict:
    # ---- system foundation: system-wide controls (no per_agent map) ----
    sys_num, sys_den = 0.0, 0.0
    gate_ok = True
    gate_reasons = []
    for cp in ALL_CONTROLS:
        r = results[cp.id]
        if not r.per_agent:  # system-wide contribution
            w = LEVEL_WEIGHTS[cp.level] * cp.weight
            sys_num += w * r.score
            sys_den += w
        # hard gate: LARGE Integrity controls must not FAIL (broken ledger)
        if cp.level == "LARGE" and cp.dimension == "Integrity" and r.score < 0.6:
            gate_ok = False
            gate_reasons.append(cp.id)
    foundation = (sys_num / sys_den) if sys_den else 1.0

    # ---- per-agent verdicts (severity-driven control outcomes) ----
    # A failure of ANY 'reject' control -> Untrusted (suspend); a failure of
    # only 'review' controls -> Under Review (audit); none -> Trusted. The
    # numeric trust score is kept as a reputation for ranking / FL gating.
    agents = {}
    alerts = []
    reputation = {}
    for aid in ctx.agent_ids():
        beh = _agent_behaviour_score(aid, results)
        trust = foundation * beh["score"] if gate_ok else 0.0
        reputation[aid] = round(trust, 4)

        verdict, failed_map = agent_verdict(results, aid, gate_ok=gate_ok)
        failed = [by_id(cid) for cid in failed_map]

        info = ctx.agents[aid]
        agents[aid] = {
            "agent_id": aid,
            "principal": info.get("principal"),
            "behaviour": info.get("behaviour"),
            "is_adversary": info.get("is_adversary"),
            "behaviour_score": round(beh["score"], 4),
            "trust_score": round(trust, 4),
            "verdict": verdict,
            "failed_controls": {cp.id: cp.severity for cp in failed},
            "weak_controls": beh["weak_controls"],
        }
        if verdict != "Trusted":
            alerts.append({"agent_id": aid, "verdict": verdict,
                           "behaviour": info.get("behaviour"),
                           "failing_controls": {cp.id: cp.severity for cp in failed}})

    verdict_counts = {"Trusted": 0, "Under Review": 0, "Untrusted": 0}
    for a in agents.values():
        verdict_counts[a["verdict"]] += 1

    return {
        "foundation_score": round(foundation, 4),
        "gate_ok": gate_ok,
        "gate_reasons": gate_reasons,
        "global_accuracy": ctx.signals.get("global_accuracy"),
        "gated_accuracy": ctx.signals.get("gated_accuracy"),
        "accuracy_uplift_from_gating": ctx.signals.get("accuracy_uplift_from_gating"),
        "dataset": ctx.signals.get("dataset"),
        "agents": agents,
        "reputation": reputation,
        "alerts": alerts,
        "verdict_counts": verdict_counts,
        "thresholds": TRUST_THRESHOLDS,
    }


def run_governance(signals: dict) -> dict:
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    verdict = score(ctx, results)
    return {
        "controls": {cid: r.as_row() for cid, r in results.items()},
        "verdict": verdict,
        "chain": ctx.chain,
    }
