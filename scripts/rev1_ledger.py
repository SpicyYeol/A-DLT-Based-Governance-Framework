# -*- coding: utf-8 -*-
"""Isolate the contribution of the ledger from that of the control logic.

The same 36 controls and the same adjudication logic are evaluated over the
same submission records under three evidence-storage conditions:

  C0  record store   — identical records and scoring logic, but no
                       tamper-evident verification (prev_hash chaining and
                       recomputed block hashes are not checked).
  C1  hash chain     — the reference testbed ledger (single append-only copy;
                       L-CI-01 link continuity + L-CI-03 block-hash integrity).
  C2  replicated     — R=4 copies of the hash chain; the audit verifies each
                       copy and compares them, reconstructing the record from a
                       strict majority of locally valid copies (no consensus
                       protocol is simulated).

After a round has been adjudicated, an insider with write access to stored
records manipulates the evidence ("post-hoc manipulation"):

  targeted   S1 launder stale (agent 5 timestamps -> fresh)
             S2 launder tamper (agent 6 data_hash := payload_hash)
             S3 launder revoked credential (delete agent 7 records)
             S4 frame honest agent 0 (timestamps back-dated by 5x SLA)
             S5 frame honest agent 2 (data_hash altered)
  random     single-field modification or deletion of a random record

Two attacker strengths: "edit" (records changed, hashes untouched) and
"rewrite" (records changed, then every subsequent block re-sealed so the copy
is internally consistent).

Per trial we record whether the audit raises an evidence-integrity alarm and
whether any re-derived verdict differs from the verdict issued at runtime.
Run:  TRUST_DATA_ROOT=<dir> python scripts/rev1_ledger.py
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trust import config
from trust import fl_simulation as fl
from trust.checks_base import Ctx
from trust.engine import evaluate_controls, score
from trust.ledger import GENESIS_HASH, Block, Ledger, sha256_hex

OUT = config.DATA_ROOT / "reports_rev1"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
R = 4                      # replicas in C2
RANDOM_TRIALS = 200        # per seed and attacker strength
SLA = config.FRESHNESS_SLA_SEC


# --------------------------------------------------------------------------
# chain helpers (same rules as Ledger.verify_chain, applied to an in-memory copy)
# --------------------------------------------------------------------------
def _block_hash(b: dict) -> str:
    return Block(**{k: b[k] for k in b if k != "block_hash"}).compute_hash()


def verify(blocks: list[dict]) -> dict:
    prev, links, intact = GENESIS_HASH, 0, 0
    for b in blocks:
        links += b["prev_hash"] == prev
        intact += _block_hash(b) == b["block_hash"]
        prev = b["block_hash"]
    n = len(blocks)
    return {"blocks": n, "valid_links": links, "total_links": n,
            "continuity": links / n if n else 1.0,
            "block_hash_integrity": intact / n if n else 1.0}


def reseal(blocks: list[dict], start: int = 0) -> None:
    """Chain-consistent rewrite from ``start``: re-index and re-seal."""
    prev = blocks[start - 1]["block_hash"] if start > 0 else GENESIS_HASH
    for i in range(start, len(blocks)):
        b = blocks[i]
        b["index"], b["prev_hash"] = i, prev
        b["block_hash"] = _block_hash(b)
        prev = b["block_hash"]


def chain_ok(c: dict) -> bool:
    return c["continuity"] == 1.0 and c["block_hash_integrity"] == 1.0


NO_CHECK = None   # C0: chain evidence unavailable -> treated as passing


def verdicts(signals: dict, blocks: list[dict], chain: dict | None) -> dict:
    if chain is None:
        n = len(blocks)
        chain = {"blocks": n, "valid_links": n, "total_links": n,
                 "continuity": 1.0, "block_hash_integrity": 1.0}
    ctx = Ctx(signals, blocks, chain, {}, config)
    res = evaluate_controls(ctx)
    sc = score(ctx, res)
    return {aid: a["verdict"] for aid, a in sc["agents"].items()}


# --------------------------------------------------------------------------
# manipulations
# --------------------------------------------------------------------------
def _targets(blocks, aid):
    return [i for i, b in enumerate(blocks) if int(b["agent_id"]) == aid]


def m_launder_stale(bl, rng):
    for i in _targets(bl, 5):
        bl[i]["timestamp"] = bl[i]["ingest_time"]
    return min(_targets(bl, 5))


def m_launder_tamper(bl, rng):
    for i in _targets(bl, 6):
        bl[i]["data_hash"] = bl[i]["payload_hash"]
    return min(_targets(bl, 6))


def m_launder_revoked(bl, rng):
    idx = _targets(bl, config.REVOKED_AGENT)
    for i in reversed(idx):
        del bl[i]
    return min(idx)


def m_frame_stale(bl, rng):
    for i in _targets(bl, 0):
        bl[i]["timestamp"] = bl[i]["ingest_time"] - 5 * SLA
    return min(_targets(bl, 0))


def m_frame_hash(bl, rng):
    for i in _targets(bl, 2):
        bl[i]["data_hash"] = sha256_hex(b"forged-%d" % i)
    return min(_targets(bl, 2))


TARGETED = {
    "S1 launder stale (agent 5)": m_launder_stale,
    "S2 launder tamper (agent 6)": m_launder_tamper,
    "S3 launder revoked (agent 7)": m_launder_revoked,
    "S4 frame honest, stale (agent 0)": m_frame_stale,
    "S5 frame honest, hash (agent 2)": m_frame_hash,
}

RANDOM_FIELDS = ["timestamp", "ingest_time", "data_hash", "payload_hash",
                 "signature", "agent_id", "principal", "scheme", "tx_type",
                 "meta.round", "delete"]


def m_random(bl, rng):
    i = rng.randrange(len(bl))
    f = rng.choice(RANDOM_FIELDS)
    b = bl[i]
    if f == "delete":
        del bl[i]
    elif f in ("timestamp", "ingest_time"):
        b[f] = b[f] + rng.choice([-1, 1]) * rng.uniform(1, 10 * SLA)
    elif f in ("data_hash", "payload_hash", "signature"):
        b[f] = sha256_hex(str(rng.random()).encode())
    elif f == "agent_id":
        b[f] = (int(b[f]) + rng.randrange(1, config.NUM_AGENTS)) % config.NUM_AGENTS
    elif f == "principal":
        b[f] = "org-x"
    elif f == "scheme":
        b[f] = "NONE"
    elif f == "tx_type":
        b[f] = "decision"
    elif f == "meta.round":
        b["meta"] = dict(b["meta"], round=b["meta"]["round"] + 1)
    return min(i, len(bl) - 1) if bl else 0, f


# --------------------------------------------------------------------------
# audits under each storage condition
# --------------------------------------------------------------------------
def apply(blocks, fn, strength, rng):
    bl = copy.deepcopy(blocks)
    out = fn(bl, rng)
    start = out[0] if isinstance(out, tuple) else out
    if strength == "rewrite" and bl:
        reseal(bl, max(0, min(start, len(bl) - 1)))
    return bl, out


def audit_c0(signals, tampered, v0):
    v = verdicts(signals, tampered, NO_CHECK)
    return {"alarm": False, "changed": v != v0, "recovered": False}


def audit_c1(signals, tampered, v0):
    c = verify(tampered)
    v = verdicts(signals, tampered, c)
    alarm = not chain_ok(c)
    return {"alarm": alarm, "changed": v != v0, "recovered": False}


def audit_c2(signals, replicas, v0):
    checks = [verify(r) for r in replicas]
    versions = [tuple(b["block_hash"] for b in r) for r in replicas]
    alarm = (not all(chain_ok(c) for c in checks)) or len(set(versions)) > 1
    valid = [i for i, c in enumerate(checks) if chain_ok(c)]
    votes = {}
    for i in valid:
        votes.setdefault(versions[i], []).append(i)
    best = max(votes.values(), key=len) if votes else []
    if len(best) > len(replicas) / 2:
        rec = replicas[best[0]]
        v = verdicts(signals, rec, verify(rec))
        status = "recovered" if v == v0 else "wrong majority"
    else:
        v, status = None, "unresolved"
    changed = (v is not None and v != v0)
    return {"alarm": alarm, "changed": changed, "recovered": status == "recovered",
            "status": status}


def run_seed(seed: int) -> dict:
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True)
    signals = naive["signals"]
    blocks = Ledger().read()
    rng = random.Random(seed)

    v0 = verdicts(signals, blocks, verify(blocks))
    # runtime adjudication is identical in all three conditions
    same = (verdicts(signals, blocks, NO_CHECK) == v0)

    rows = []
    for name, fn in TARGETED.items():
        for strength in ("edit", "rewrite"):
            bl, _ = apply(blocks, fn, strength, rng)
            r0 = audit_c0(signals, bl, v0)
            r1 = audit_c1(signals, bl, v0)
            reps = [copy.deepcopy(blocks) for _ in range(R)]
            reps[0] = bl                                   # k = 1 compromised copy
            r2 = audit_c2(signals, reps, v0)
            # boundary: all R copies rewritten consistently
            r2_all = audit_c2(signals, [copy.deepcopy(bl) for _ in range(R)], v0)
            # boundary: k = 2 of 4 copies
            r2_half = audit_c2(signals, [bl, copy.deepcopy(bl)] +
                               [copy.deepcopy(blocks) for _ in range(R - 2)], v0)
            rows.append({"seed": seed, "scenario": name, "strength": strength,
                         "C0": r0, "C1": r1, "C2_k1": r2, "C2_k2": r2_half,
                         "C2_kR": r2_all})

    rand = []
    for strength in ("edit", "rewrite"):
        for _ in range(RANDOM_TRIALS):
            bl, (pos, field) = apply(blocks, m_random, strength, rng)
            r0 = audit_c0(signals, bl, v0)
            r1 = audit_c1(signals, bl, v0)
            reps = [copy.deepcopy(blocks) for _ in range(R)]
            reps[0] = bl
            r2 = audit_c2(signals, reps, v0)
            rand.append({"seed": seed, "strength": strength, "field": field,
                         "C0": r0, "C1": r1, "C2_k1": r2})

    # storage / verification cost per condition
    raw = sum(len(json.dumps({k: v for k, v in b.items()
                              if k not in ("prev_hash", "block_hash")})) + 1 for b in blocks)
    chained = sum(len(json.dumps(b)) + 1 for b in blocks)
    t = time.perf_counter()
    for _ in range(200):
        verify(blocks)
    t_verify = (time.perf_counter() - t) / 200
    return {"seed": seed, "n_blocks": len(blocks), "runtime_verdicts_identical": same,
            "runtime_verdicts": v0, "targeted": rows, "random": rand,
            "bytes_C0": raw, "bytes_C1": chained, "bytes_C2": chained * R,
            "t_verify_s": t_verify}


def main() -> int:
    out = [run_seed(s) for s in SEEDS]
    json.dump(out, open(OUT / "rev1_ledger.json", "w"), indent=1, default=str)
    print("done", [o["seed"] for o in out])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
