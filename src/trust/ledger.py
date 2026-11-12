"""Simulated DLT Trust Registry.

An append-only, hash-chained JSONL ledger. Each block records the tamper-evident
audit trail the framework requires:

    { agent_id, data_hash, signature, timestamp, prev_hash, block_hash, ... }

This is not a real blockchain — it is a faithful *governance-layer* stand-in so
the LARGE/Consensus and MEDIUM/Traceability and SMALL/Message-Integrity control
points have something concrete and verifiable to evaluate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config

GENESIS_HASH = "0" * 64


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def agent_key(agent_id: int) -> bytes:
    """Deterministic per-agent HMAC key material (stands in for a signing key)."""
    return hashlib.sha256(f"agent-key-{agent_id}".encode()).digest()


def sign(agent_id: int, payload_hash: str) -> str:
    return hmac.new(agent_key(agent_id), payload_hash.encode(), hashlib.sha256).hexdigest()


def verify(agent_id: int, payload_hash: str, signature: str) -> bool:
    return hmac.compare_digest(sign(agent_id, payload_hash), signature)


@dataclass
class Block:
    index: int
    agent_id: int
    principal: str
    tx_type: str                 # e.g. "data_packet" | "model_update" | "decision"
    data_hash: str               # hash the submitter *claims*
    payload_hash: str            # hash recomputed from the recorded payload
    signature: str
    scheme: str
    timestamp: float             # submitter-supplied
    ingest_time: float           # registry receive time
    prev_hash: str
    meta: dict = field(default_factory=dict)
    block_hash: str = ""

    def compute_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "block_hash"}
        return sha256_hex(json.dumps(body, sort_keys=True).encode())

    def seal(self) -> "Block":
        self.block_hash = self.compute_hash()
        return self


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or (config.LEDGER_DIR / "trust_registry.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def _last_hash(self) -> str:
        last = GENESIS_HASH
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        last = json.loads(line)["block_hash"]
        except FileNotFoundError:            # tolerate a concurrent reset
            pass
        return last

    def _count(self) -> int:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except FileNotFoundError:            # tolerate a concurrent reset
            return 0

    def append(
        self,
        agent_id: int,
        principal: str,
        tx_type: str,
        payload: bytes,
        *,
        claimed_hash: Optional[str] = None,
        signature: Optional[str] = None,
        scheme: str = config.SIGNING_SCHEME,
        timestamp: Optional[float] = None,
        ingest_time: Optional[float] = None,
        meta: Optional[dict] = None,
    ) -> Block:
        """Append a block. ``claimed_hash``/``signature`` let a simulated
        adversary submit values that disagree with the true payload so integrity
        controls have something to catch."""
        payload_hash = sha256_hex(payload)
        data_hash = claimed_hash if claimed_hash is not None else payload_hash
        ts = timestamp if timestamp is not None else time.time()
        ing = ingest_time if ingest_time is not None else time.time()
        sig = signature if signature is not None else sign(agent_id, data_hash)
        blk = Block(
            index=self._count(),
            agent_id=agent_id,
            principal=principal,
            tx_type=tx_type,
            data_hash=data_hash,
            payload_hash=payload_hash,
            signature=sig,
            scheme=scheme,
            timestamp=ts,
            ingest_time=ing,
            prev_hash=self._last_hash(),
            meta=meta or {},
        ).seal()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(blk)) + "\n")
        return blk

    def read(self) -> list[dict]:
        out = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except FileNotFoundError:
            return []
        return out

    def verify_chain(self) -> dict:
        """LARGE/Consensus-Integrity primitive: check hash-chain continuity."""
        blocks = self.read()
        valid_links = 0
        total_links = 0
        prev = GENESIS_HASH
        recompute_ok = 0
        for b in blocks:
            total_links += 1
            if b["prev_hash"] == prev:
                valid_links += 1
            # verify the block hash itself was not rewritten
            tmp = Block(**{k: b[k] for k in b if k != "block_hash"})
            if tmp.compute_hash() == b["block_hash"]:
                recompute_ok += 1
            prev = b["block_hash"]
        return {
            "blocks": len(blocks),
            "valid_links": valid_links,
            "total_links": total_links,
            "continuity": (valid_links / total_links) if total_links else 1.0,
            "block_hash_integrity": (recompute_ok / total_links) if total_links else 1.0,
        }
