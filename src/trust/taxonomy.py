"""
Single source of truth for the DLT Agent-Trustworthiness Governance Framework.

Encodes the 3-Tier x 12-Category x 36-Sub-Category structure and *fills in*
the two columns that the taxonomy itself leaves empty:

    Level (3) | Category (12) | Sub-Category (36) | Detailed Category | Control points

Each sub-category gets one concrete, testable control point with a metric and a
Clause-8 style threshold. This module is deliberately dependency-free so it can
be imported cheaply at DAG-parse time.

Design notes
------------
* ``level``     : LARGE / MEDIUM / SMALL  (Clause 6 trust hierarchy)
* ``dimension`` : Integrity / Authenticity / Availability / Accountability
                  (the four governance-engine dimensions)
* ``method``    : "data"        -> computed live from the open datasets / ledger
                  "attestation" -> governance/policy control, evidence-attested
* ``orient``    : "general" (any AI/IoT/identity agent),
                  "ML"      (classical ML oriented),
                  "DL"      (deep-learning / federated-learning oriented)
                  -> answers the requirement: target is AI in general, but
                     DL/ML-oriented items are proposed explicitly.
* ``check``     : dotted name of the callable that evaluates the control point,
                  or None for pure attestation controls.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

# Clause-6 level weights used when rolling sub-scores up to an overall verdict.
# LARGE is the structural prerequisite ("why the system can exist") and acts as
# a gate; MEDIUM and SMALL contribute to the per-agent trust score.
LEVEL_WEIGHTS = {"LARGE": 0.40, "MEDIUM": 0.35, "SMALL": 0.25}

# Clause-8 classification thresholds -> the three control outcomes.
TRUST_THRESHOLDS = {"trusted": 0.80, "under_review": 0.60}  # below -> untrusted


@dataclass
class ControlPoint:
    id: str
    level: str            # LARGE | MEDIUM | SMALL
    category: str         # one of the 12
    subcategory: str      # one of the 36
    detailed_category: str
    control_point: str
    metric: str
    threshold: str
    dimension: str        # Integrity | Authenticity | Availability | Accountability
    method: str           # data | attestation
    orient: str           # general | ML | DL
    check: Optional[str] = None   # dotted path to evaluator, e.g. "small.freshness"
    weight: float = 1.0
    # governance severity of a FAILURE of this control -> drives the verdict:
    #   reject -> Untrusted (suspend agent), review -> Under Review (audit),
    #   info   -> advisory only (affects the numeric reputation, not the tier)
    severity: str = "info"

    def as_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# LARGE - System-wide trust foundation (Clause 6-1)
# ---------------------------------------------------------------------------
_LARGE = [
    ControlPoint(
        "L-IA-01", "LARGE", "Identity & Authentication", "Agent Identity",
        "Unique agent identifier registry",
        "Every contributing agent presents a unique, resolvable DID/identifier "
        "bound to a public key before any submission is accepted.",
        "unique_identity_ratio = distinct_valid_ids / total_agents",
        ">= 1.0 (no unknown/duplicate identities)",
        "Authenticity", "data", "general", check="large.agent_identity"),
    ControlPoint(
        "L-IA-02", "LARGE", "Identity & Authentication", "Organizational Identity",
        "Principal/organization binding",
        "Each agent identity is bound to an accountable principal (org) record.",
        "principal_binding_ratio",
        ">= 0.95", "Accountability", "attestation", "general"),
    ControlPoint(
        "L-IA-03", "LARGE", "Identity & Authentication", "Identity Lifecycle",
        "Enrollment / rotation / revocation policy",
        "Key rotation and revocation procedures exist and revoked agents are "
        "rejected by the registry.",
        "revoked_agent_rejection_rate",
        "= 1.0 for revoked agents", "Authenticity", "data", "general",
        check="large.identity_lifecycle", severity="reject"),

    ControlPoint(
        "L-CI-01", "LARGE", "Consensus Integrity", "Consensus Mechanism",
        "Ledger append consensus / hash-chain continuity",
        "The trust registry forms an unbroken hash chain; no block links to a "
        "wrong predecessor hash.",
        "chain_continuity = valid_links / total_links",
        "= 1.0", "Integrity", "data", "general", check="large.consensus_chain", severity="reject"),
    ControlPoint(
        "L-CI-02", "LARGE", "Consensus Integrity", "Node Governance",
        "Validator/node admission control",
        "Only admitted nodes may write; quorum/fault-tolerance policy defined.",
        "unauthorized_write_attempts",
        "= 0", "Accountability", "attestation", "general"),
    ControlPoint(
        "L-CI-03", "LARGE", "Consensus Integrity", "Finality Assurance",
        "Block finality confirmation",
        "Recorded blocks reach finality (are not reorganised/rewritten).",
        "reorg_events",
        "= 0", "Integrity", "data", "general", check="large.finality"),

    ControlPoint(
        "L-GA-01", "LARGE", "Governance Accountability", "Policy Management",
        "Documented governance policy set",
        "Trust policies, roles and thresholds (this taxonomy) are versioned and "
        "available to auditors.",
        "policy_coverage = documented_controls / total_controls",
        ">= 0.90", "Accountability", "attestation", "general"),
    ControlPoint(
        "L-GA-02", "LARGE", "Governance Accountability", "Decision Transparency",
        "Auditable decision records",
        "Every Trusted/Under-Review/Untrusted decision is logged with the "
        "evidence that produced it.",
        "decision_traceability_ratio",
        "= 1.0", "Accountability", "data", "general",
        check="large.decision_transparency"),
    ControlPoint(
        "L-GA-03", "LARGE", "Governance Accountability", "Regulatory Compliance",
        "Alignment with applicable regulation",
        "Mapping to applicable regulatory/legal requirements is maintained.",
        "regulatory_mapping_completeness",
        ">= 0.90", "Accountability", "attestation", "general"),

    ControlPoint(
        "L-CT-01", "LARGE", "Cross-domain Trust Federation", "Trust Anchor Management",
        "Trust anchor / root-of-trust inventory",
        "Accepted trust anchors (CA / root keys) are enumerated and their "
        "fingerprints pinned.",
        "pinned_anchor_ratio",
        ">= 0.95", "Authenticity", "attestation", "general"),
    ControlPoint(
        "L-CT-02", "LARGE", "Cross-domain Trust Federation", "Interoperability",
        "Cross-domain schema/credential interop",
        "Credentials and data schemas validate against the shared spec so "
        "external-domain agents can be evaluated.",
        "schema_conformance_ratio",
        ">= 0.98", "Integrity", "data", "general", check="large.interoperability"),
    ControlPoint(
        "L-CT-03", "LARGE", "Cross-domain Trust Federation", "Mutual Recognition",
        "Federated trust agreement recognition",
        "Trust decisions from recognised peer domains are honoured per policy.",
        "recognised_domain_ratio",
        ">= 0.90", "Accountability", "attestation", "general"),
]

# ---------------------------------------------------------------------------
# MEDIUM - Agent / Transaction level trust (Clause 6-2)
# ---------------------------------------------------------------------------
_MEDIUM = [
    ControlPoint(
        "M-BC-01", "MEDIUM", "Behavioral Consistency", "Activity Monitoring",
        "Model-update anomaly detection (FL)",
        "Each agent's model update magnitude is compared to the cohort; "
        "outliers (poisoning / scaling attacks) are flagged.",
        "update_norm_zscore",
        "|z| <= 3.0", "Integrity", "data", "DL",
        check="medium.update_anomaly", severity="reject"),
    ControlPoint(
        "M-BC-02", "MEDIUM", "Behavioral Consistency", "Historical Analysis",
        "Cross-round contribution stability",
        "An agent's per-round accuracy contribution should not swing erratically "
        "or trend negative across rounds.",
        "contribution_stability = 1 - std(round_contribution)",
        ">= 0.5", "Availability", "data", "ML",
        check="medium.historical_stability"),
    ControlPoint(
        "M-BC-03", "MEDIUM", "Behavioral Consistency", "Penalty & Recovery",
        "Reputation penalty / recovery policy",
        "Flagged agents receive score penalties and a defined recovery path.",
        "penalty_application_rate for flagged agents",
        "= 1.0", "Accountability", "data", "general",
        check="medium.penalty_recovery"),

    ControlPoint(
        "M-CV-01", "MEDIUM", "Credential Validity", "Issuance Control",
        "Signed-submission credential check",
        "Every submission carries a credential/signature issued to the agent; "
        "unsigned submissions are rejected.",
        "signed_submission_ratio",
        "= 1.0", "Authenticity", "data", "general",
        check="medium.issuance_control"),
    ControlPoint(
        "M-CV-02", "MEDIUM", "Credential Validity", "Lifecycle Management",
        "Credential validity window",
        "Credential is within its validity window (not expired) at submission.",
        "valid_window_ratio",
        ">= 0.99", "Authenticity", "data", "general",
        check="medium.credential_lifecycle", severity="reject"),
    ControlPoint(
        "M-CV-03", "MEDIUM", "Credential Validity", "Verification Process",
        "Signature verification pass rate",
        "Submission signature verifies against the agent's registered key.",
        "signature_verify_pass_rate",
        "= 1.0", "Authenticity", "data", "general",
        check="medium.verification_process", severity="reject"),

    ControlPoint(
        "M-TT-01", "MEDIUM", "Transaction Traceability", "Audit Trail",
        "Complete per-transaction ledger record",
        "Each contribution is recorded as a ledger block with agent id, hash, "
        "signature and timestamp.",
        "ledger_coverage = recorded_tx / total_tx",
        "= 1.0", "Accountability", "data", "general",
        check="medium.audit_trail", severity="review"),
    ControlPoint(
        "M-TT-02", "MEDIUM", "Transaction Traceability", "Provenance Tracking",
        "Data provenance / lineage",
        "Data packet provenance (source shard, transform) is recoverable from "
        "the ledger.",
        "provenance_completeness",
        ">= 0.95", "Integrity", "data", "general",
        check="medium.provenance"),
    ControlPoint(
        "M-TT-03", "MEDIUM", "Transaction Traceability", "Accountability Mapping",
        "Transaction-to-principal mapping",
        "Every transaction maps back to an accountable principal.",
        "accountability_mapping_ratio",
        "= 1.0", "Accountability", "data", "general",
        check="medium.accountability_mapping"),

    ControlPoint(
        "M-SC-01", "MEDIUM", "Smart Contract Compliance", "Code Assurance",
        "Aggregation-rule (contract) integrity",
        "The aggregation rule / smart-contract code hash matches the approved "
        "reference hash.",
        "contract_hash_match",
        "= approved", "Integrity", "data", "general",
        check="medium.code_assurance"),
    ControlPoint(
        "M-SC-02", "MEDIUM", "Smart Contract Compliance", "Execution Monitoring",
        "Aggregation execution bounds",
        "Contract execution stays within declared bounds (e.g. contribution "
        "weight caps) — detects Byzantine over-weighting.",
        "within_bounds_ratio",
        "= 1.0", "Integrity", "data", "DL",
        check="medium.execution_monitoring", severity="reject"),
    ControlPoint(
        "M-SC-03", "MEDIUM", "Smart Contract Compliance", "Update & Versioning",
        "Contract/model version control",
        "Global model + contract versions increment monotonically and are "
        "recorded.",
        "monotonic_version_ratio",
        "= 1.0", "Integrity", "data", "general",
        check="medium.versioning"),
]

# ---------------------------------------------------------------------------
# SMALL - Trust in detailed operational & technical levels (Clause 6-3)
# ---------------------------------------------------------------------------
_SMALL = [
    ControlPoint(
        "S-DF-01", "SMALL", "Data Freshness", "Timestamp Management",
        "Trusted timestamp on each packet",
        "Each packet carries a monotonic, non-future timestamp.",
        "valid_timestamp_ratio",
        ">= 0.99", "Integrity", "data", "general", check="small.timestamp", severity="review"),
    ControlPoint(
        "S-DF-02", "SMALL", "Data Freshness", "Update Frequency",
        "Staleness / heartbeat window",
        "Packet age at ingestion is within the freshness SLA (detects stale / "
        "replayed / lazy-agent data).",
        "fresh_ratio (age <= SLA)",
        ">= 0.95", "Availability", "data", "general", check="small.freshness",
        weight=2.5, severity="review"),
    ControlPoint(
        "S-DF-03", "SMALL", "Data Freshness", "Consistency Check",
        "Feature-distribution drift check (ISO/IEC 5259)",
        "Submitted data distribution is consistent with the expected profile "
        "(KL / PSI drift) — catches corrupted or off-distribution shards.",
        "distribution_drift (KL divergence)",
        "<= 0.20", "Integrity", "data", "ML", check="small.consistency",
        weight=3.0, severity="review"),

    ControlPoint(
        "S-MI-01", "SMALL", "Message Integrity", "Signing Standard",
        "Approved signature algorithm",
        "Signatures use the approved algorithm/scheme (HMAC-SHA256 in this "
        "testbed) — no weak/absent scheme.",
        "approved_scheme_ratio",
        "= 1.0", "Authenticity", "data", "general", check="small.signing_standard", severity="review"),
    ControlPoint(
        "S-MI-02", "SMALL", "Message Integrity", "Hash Management",
        "Payload hash verification",
        "Recomputed payload hash matches the recorded hash (tamper detection).",
        "hash_match_ratio",
        "= 1.0", "Integrity", "data", "general", check="small.hash_management", severity="reject"),
    ControlPoint(
        "S-MI-03", "SMALL", "Message Integrity", "Transmission Security",
        "Transport confidentiality/integrity",
        "Packets are transmitted over an integrity-protected channel.",
        "secure_channel_ratio",
        "= 1.0", "Integrity", "attestation", "general"),

    ControlPoint(
        "S-AL-01", "SMALL", "Access Logging", "Log Classification",
        "Structured, classified access logs",
        "Every access/submission event is logged with a severity/type class.",
        "classified_log_ratio",
        ">= 0.98", "Accountability", "data", "general", check="small.log_classification"),
    ControlPoint(
        "S-AL-02", "SMALL", "Access Logging", "Retention & Storage",
        "Immutable log retention",
        "Logs are retained append-only for the required window.",
        "retention_conformance",
        ">= 0.95", "Accountability", "attestation", "general"),
    ControlPoint(
        "S-AL-03", "SMALL", "Access Logging", "Monitoring & Alert",
        "Anomaly alerting coverage",
        "Detected anomalies raise an alert event in the log.",
        "alerted_anomaly_ratio",
        "= 1.0", "Availability", "data", "general", check="small.monitoring_alert"),

    ControlPoint(
        "S-RS-01", "SMALL", "Agent Reputation Score", "Score Calculation",
        "Composite reputation computation",
        "Per-agent reputation is computed from behavioural + data-quality + "
        "integrity signals.",
        "reputation_score in [0,1]",
        "computed for 100% of agents", "Accountability", "data", "ML",
        check="small.score_calculation"),
    ControlPoint(
        "S-RS-02", "SMALL", "Agent Reputation Score", "Score Maintenance",
        "Reputation decay / update over rounds",
        "Reputation is updated each round with decay so past behaviour is "
        "weighted.",
        "score_update_coverage",
        "= 1.0", "Accountability", "data", "ML", check="small.score_maintenance"),
    ControlPoint(
        "S-RS-03", "SMALL", "Agent Reputation Score", "Score Application",
        "Reputation-weighted aggregation gating",
        "Low-reputation agents are down-weighted or excluded from aggregation.",
        "gating_applied_ratio for low-rep agents",
        "= 1.0", "Integrity", "data", "DL", check="small.score_application"),
]

ALL_CONTROLS = _LARGE + _MEDIUM + _SMALL

# Sanity: the taxonomy is exactly 3 x 12 x 36.
assert len({c.level for c in ALL_CONTROLS}) == 3, "expected 3 levels"
assert len({c.category for c in ALL_CONTROLS}) == 12, "expected 12 categories"
assert len(ALL_CONTROLS) == 36, f"expected 36 sub-categories, got {len(ALL_CONTROLS)}"


def by_level(level: str) -> list[ControlPoint]:
    return [c for c in ALL_CONTROLS if c.level == level]


def by_id(cid: str) -> ControlPoint:
    for c in ALL_CONTROLS:
        if c.id == cid:
            return c
    raise KeyError(cid)


CATEGORIES = {}
for _c in ALL_CONTROLS:
    CATEGORIES.setdefault((_c.level, _c.category), []).append(_c.id)
