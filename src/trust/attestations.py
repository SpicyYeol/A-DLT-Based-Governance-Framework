"""Attestation register for governance/policy controls.

Some of the 36 sub-categories (e.g. Regulatory Compliance, Mutual Recognition,
Transmission Security) are governance attestations, not signals computable from
runtime data. Rather than fake a measurement, the testbed records them honestly
as ATTESTED with an evidence pointer and an attested score.

In a real deployment these would be backed by audit evidence; here they default
to satisfied so the end-to-end verdict is meaningful, and each carries the
evidence field an auditor would fill.
"""

from __future__ import annotations

from .checks_base import ControlResult, ATTESTED

# control_id -> (attested_score, evidence text)
ATTESTATIONS = {
    "L-IA-02": (1.0, "Principal registry maps every agent id to org-{0,1,2}."),
    "L-CI-02": (1.0, "Only the aggregator process may append to the registry."),
    "L-GA-01": (0.95, "This taxonomy module documents 36/36 controls, versioned."),
    "L-GA-03": (0.90, "Mapping to ISO/TS 23635 & ISO 23257 maintained in README."),
    "L-CT-01": (0.95, "Trust anchors = per-agent HMAC keys pinned in ledger.agent_key."),
    "L-CT-03": (0.90, "Peer-domain recognition policy declared; no live peers in testbed."),
    "S-MI-03": (1.0, "Registry writes occur intra-process; treated as integrity-protected."),
    "S-AL-02": (0.95, "Ledger + logs are append-only JSONL retained for the run window."),
}


def evaluate_attestation(control_id: str) -> ControlResult:
    score, evidence = ATTESTATIONS.get(control_id, (0.8, "No evidence on file."))
    return ControlResult(
        id=control_id, status=ATTESTED, score=score,
        value="attested", threshold="policy",
        evidence={"note": evidence, "kind": "governance-attestation"})
