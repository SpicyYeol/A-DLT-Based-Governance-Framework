# -*- coding: utf-8 -*-
"""Scalability to larger, higher-frequency
agent populations.

Part 1  Participant sweep N = 8 ... 1024 (3 rounds, full MNIST so that every
        agent keeps a non-trivial shard). Threat roles are assigned
        proportionally (agent_id mod 8 follows the primary profile, including
        one revoked credential per group of eight), so the problematic share
        stays at 62.5% at every N. Records governance-check latency,
        full chain-verification latency, ledger size, and detection counts.

Part 2  Throughput micro-benchmarks for high-frequency operation on one core:
        sealed append, HMAC verification, incremental (per-new-block) chain
        verification, and full re-verification.

Only local computation is measured; no consensus protocol or network is
simulated: no consensus protocol is run.
Run:  TRUST_DATA_ROOT=<dir> TRUST_IMAGE_CAP=70000 python scripts/rev1_scale.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sklearn

from trust import config
from trust import fl_simulation as fl
from trust import ledger as L
from trust.checks_base import ControlResult, PASS, WARN, status_from_score
from trust.controls import large, medium
from trust.engine import build_ctx, evaluate_controls
from trust.experiments import _conf, _flagged_set

OUT = config.DATA_ROOT / "reports_rev1"
OUT.mkdir(parents=True, exist_ok=True)

NS = [8, 16, 32, 64, 128, 256, 512, 1024]
SEEDS = [42, 43, 44]
ROLE = {1: "label_flip", 3: "scale_attack", 5: "stale", 6: "tamper"}   # primary
REVOKED_SLOT = 7


# --------------------------------------------------------------------------
# 1) Ledger append keeps the chain head in memory (the reference append
#    re-reads the whole file to find the head, which is O(B) per block and
#    would dominate large runs). Block contents and hashing are unchanged.
# --------------------------------------------------------------------------
class HeadCachedLedger(L.Ledger):
    _head: dict = {}

    def reset(self):
        super().reset()
        HeadCachedLedger._head[str(self.path)] = (L.GENESIS_HASH, 0)

    def _last_hash(self):
        h = HeadCachedLedger._head.get(str(self.path))
        return h[0] if h else super()._last_hash()

    def _count(self):
        h = HeadCachedLedger._head.get(str(self.path))
        return h[1] if h else super()._count()

    def append(self, *a, **kw):
        blk = super().append(*a, **kw)
        HeadCachedLedger._head[str(self.path)] = (blk.block_hash, blk.index + 1)
        return blk


fl.Ledger = HeadCachedLedger


# --------------------------------------------------------------------------
# 2) Revocation controls accept a set of revoked identities (the reference
#    controls assume a single revoked agent). Decision rules are unchanged.
# --------------------------------------------------------------------------
def _revoked(ctx):
    r = ctx.signals.get("revoked_agent")
    if r is None:
        return set()
    return set(r) if isinstance(r, (list, tuple, set)) else {int(r)}


def identity_lifecycle(ctx):  # L-IA-03
    rev = _revoked(ctx)
    offenders = {int(b["agent_id"]) for b in ctx.ledger_blocks
                 if int(b["agent_id"]) in rev}
    detected = bool(offenders)
    return ControlResult("L-IA-03", WARN if detected else PASS,
                         0.9 if detected else 1.0,
                         value=f"{len(offenders)} revoked identities wrote",
                         per_agent={a: 0.0 for a in offenders})


def credential_lifecycle(ctx):  # M-CV-02
    rev = _revoked(ctx)
    per_agent, invalid = {}, 0
    for b in ctx.ledger_blocks:
        aid = int(b["agent_id"])
        if aid in rev:
            per_agent[aid] = 0.0
            invalid += 1
    ratio = 1.0 - invalid / max(len(ctx.ledger_blocks), 1)
    return ControlResult("M-CV-02", status_from_score(ratio, ok=0.99), ratio,
                         value=round(ratio, 4), per_agent=per_agent)


large.identity_lifecycle = identity_lifecycle
medium.credential_lifecycle = credential_lifecycle


def proportional_profile(n):
    prof = {a: ROLE[a % 8] for a in range(n) if a % 8 in ROLE}
    revoked = [a for a in range(n) if a % 8 == REVOKED_SLOT]
    return prof, revoked


def measure(n, seed):
    config.NUM_AGENTS = n
    prof, revoked = proportional_profile(n)
    config.REVOKED_AGENT = revoked
    prep = fl.prepare(seed)

    t0 = time.perf_counter()
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=prof)
    t_round_total = time.perf_counter() - t0          # training + ledger writes

    t0 = time.perf_counter()
    ctx = build_ctx(naive["signals"])
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    flagged = _flagged_set(results, ids)
    t_gov = time.perf_counter() - t0

    led = L.Ledger()
    t0 = time.perf_counter()
    chain = led.verify_chain()
    t_chain = time.perf_counter() - t0

    gt = set(prof) | set(revoked)
    det = _conf(flagged, ids, gt)
    return {"N": n, "seed": seed, "blocks": chain["blocks"],
            "ledger_bytes": led.path.stat().st_size,
            "t_train_and_record_s": t_round_total, "t_governance_s": t_gov,
            "t_chain_verify_s": t_chain,
            "chain_ok": chain["continuity"] == 1.0,
            **{k: det[k] for k in ("TP", "FP", "FN", "TN")}}


def bench(nblocks=20000):
    """Single-core throughput of the ledger primitives."""
    import hashlib, hmac
    blocks, prev = [], L.GENESIS_HASH
    t0 = time.perf_counter()
    for i in range(nblocks):
        payload = f"agent{i % 1024}-round{i // 1024}".encode()
        ph = L.sha256_hex(payload)
        b = L.Block(index=i, agent_id=i % 1024, principal=f"org-{i % 3}",
                    tx_type="data_packet", data_hash=ph, payload_hash=ph,
                    signature=L.sign(i % 1024, ph), scheme="HMAC-SHA256",
                    timestamp=1.0, ingest_time=1.0, prev_hash=prev,
                    meta={"round": i // 1024, "logged": True}).seal()
        prev = b.block_hash
        blocks.append(b)
    t_append = time.perf_counter() - t0

    t0 = time.perf_counter()
    ok = sum(L.verify(b.agent_id, b.data_hash, b.signature) for b in blocks)
    t_hmac = time.perf_counter() - t0

    # incremental verification: each new block checked against the cached head
    t0 = time.perf_counter()
    head, good = L.GENESIS_HASH, 0
    for b in blocks:
        good += (b.prev_hash == head) and (b.compute_hash() == b.block_hash)
        head = b.block_hash
    t_incr = time.perf_counter() - t0
    assert ok == nblocks and good == nblocks
    return {"blocks": nblocks,
            "append_blocks_per_s": nblocks / t_append,
            "hmac_verify_per_s": nblocks / t_hmac,
            "incremental_verify_blocks_per_s": nblocks / t_incr}


def main() -> int:
    env = {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
           "cpu": platform.processor(), "image_cap": config.IMAGE_SAMPLE_CAP,
           "rounds": config.NUM_ROUNDS}
    rows = []
    for n in NS:
        for s in SEEDS:
            r = measure(n, s)
            rows.append(r)
            print(json.dumps(r), flush=True)
    b = bench()
    print(json.dumps(b), flush=True)
    summary = []
    for n in NS:
        rr = [r for r in rows if r["N"] == n]
        summary.append({
            "N": n, "blocks": rr[0]["blocks"],
            "ledger_kb": rr[0]["ledger_bytes"] / 1024,
            "t_train_and_record_s": st.mean(r["t_train_and_record_s"] for r in rr),
            "t_governance_ms": 1e3 * st.mean(r["t_governance_s"] for r in rr),
            "t_governance_ms_sd": 1e3 * st.pstdev([r["t_governance_s"] for r in rr]),
            "t_chain_ms": 1e3 * st.mean(r["t_chain_verify_s"] for r in rr),
            **{k: sum(r[k] for r in rr) for k in ("TP", "FP", "FN", "TN")}})
    json.dump({"env": env, "rows": rows, "summary": summary, "bench": b},
              open(OUT / "rev1_scale.json", "w"), indent=1)
    for s_ in summary:
        print(json.dumps(s_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
