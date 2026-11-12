"""Common result type and evaluation context for control-point checks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# status vocabulary
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
ATTESTED = "ATTESTED"       # governance/policy control, evidence-attested
NA = "NOT_APPLICABLE"


@dataclass
class ControlResult:
    id: str
    status: str
    score: float                       # 0..1 system-level score for this control
    value: Any = None                  # measured metric value
    threshold: str = ""
    evidence: dict = field(default_factory=dict)
    per_agent: dict = field(default_factory=dict)   # agent_id -> subscore 0..1

    def as_row(self) -> dict:
        d = asdict(self)
        d["per_agent"] = {str(k): v for k, v in self.per_agent.items()}
        return d


def status_from_score(score: float, warn: float = 0.8, ok: float = 0.999) -> str:
    if score >= ok:
        return PASS
    if score >= warn:
        return WARN
    return FAIL


class Ctx:
    """Everything the checks need, assembled once per governance run."""

    def __init__(self, signals: dict, ledger_blocks: list[dict],
                 chain: dict, tabular: dict, cfg):
        self.signals = signals
        self.ledger_blocks = ledger_blocks
        self.chain = chain
        self.tabular = tabular
        self.cfg = cfg
        self.agents = signals.get("agents", {})
        # normalise agent keys to int
        self.agents = {int(k): v for k, v in self.agents.items()}

    def agent_ids(self) -> list[int]:
        return sorted(self.agents.keys())

    def blocks_of(self, agent_id: int) -> list[dict]:
        return [b for b in self.ledger_blocks if int(b["agent_id"]) == agent_id]
