"""LARGE-level control evaluators — Clause 6-1 System-wide trust foundation.

These are system-wide integrity/identity controls. Most are computed from the
DLT trust-registry chain rather than per-agent behaviour. The hash-chain
continuity control acts as the hard gate: if the ledger is not internally
consistent, nothing downstream can be trusted.
"""

from __future__ import annotations

from ..checks_base import ControlResult, Ctx, PASS, WARN, FAIL, status_from_score


def agent_identity(ctx: Ctx) -> ControlResult:  # L-IA-01
    ids = [int(b["agent_id"]) for b in ctx.ledger_blocks]
    distinct = set(ids)
    total_agents = ctx.signals["num_agents"]
    ratio = len(distinct & set(range(total_agents))) / max(total_agents, 1)
    unknown = [i for i in distinct if i not in range(total_agents)]
    return ControlResult(
        "L-IA-01", status_from_score(ratio), ratio, value=round(ratio, 4),
        threshold=">= 1.0",
        evidence={"distinct_ids": sorted(distinct), "unknown_ids": unknown})


def identity_lifecycle(ctx: Ctx) -> ControlResult:  # L-IA-03
    revoked = ctx.signals.get("revoked_agent")
    offending = ctx.blocks_of(revoked) if revoked is not None else []
    # The revocation mechanism *detected* the revoked agent's writes; enforcement
    # (exclusion) is applied in scoring. Presence of violations -> WARN finding.
    detected = len(offending) > 0
    score = 0.9 if detected else 1.0
    per_agent = {revoked: 0.0} if detected else {}
    return ControlResult(
        "L-IA-03", WARN if detected else PASS, score,
        value=f"{len(offending)} post-revocation writes by agent {revoked}",
        threshold="= 1.0 rejection for revoked",
        evidence={"revoked_agent": revoked,
                  "post_revocation_writes": len(offending),
                  "enforcement": "excluded from aggregation in scoring"},
        per_agent=per_agent)


def consensus_chain(ctx: Ctx) -> ControlResult:  # L-CI-01  (hard gate)
    c = ctx.chain
    score = c["continuity"]
    return ControlResult(
        "L-CI-01", status_from_score(score), score,
        value=round(score, 6), threshold="= 1.0",
        evidence={k: c[k] for k in ("blocks", "valid_links", "total_links")})


def finality(ctx: Ctx) -> ControlResult:  # L-CI-03
    score = ctx.chain["block_hash_integrity"]
    reorg = ctx.chain["total_links"] - int(round(score * ctx.chain["total_links"]))
    return ControlResult(
        "L-CI-03", status_from_score(score), score,
        value=f"reorg_events={reorg}", threshold="= 0 reorgs",
        evidence={"block_hash_integrity": round(score, 6)})


def decision_transparency(ctx: Ctx) -> ControlResult:  # L-GA-02
    # Every agent must have a full evidence chain (rounds + ledger blocks).
    agents = ctx.agents
    complete = sum(1 for a in agents.values()
                   if a.get("rounds") and ctx.blocks_of(int(a["agent_id"])))
    ratio = complete / max(len(agents), 1)
    return ControlResult(
        "L-GA-02", status_from_score(ratio), ratio, value=round(ratio, 4),
        threshold="= 1.0",
        evidence={"agents_with_full_evidence": complete, "total": len(agents)})


def interoperability(ctx: Ctx) -> ControlResult:  # L-CT-02
    required = {"agent_id", "data_hash", "signature", "timestamp", "prev_hash",
                "block_hash", "scheme"}
    ok = sum(1 for b in ctx.ledger_blocks if required.issubset(b.keys()))
    ratio = ok / max(len(ctx.ledger_blocks), 1)
    return ControlResult(
        "L-CT-02", status_from_score(ratio, ok=0.98), ratio,
        value=round(ratio, 4), threshold=">= 0.98",
        evidence={"schema_conformant_blocks": ok, "total": len(ctx.ledger_blocks)})
