# -*- coding: utf-8 -*-
"""E6 (P0-2): which auditability property actually distinguishes the framework?

The online-adjudication comparison (Table 11 of the paper) reported a verdict
reconstruction rate of 1.000 for the conventional
baseline as well as for governance, so reconstruction alone cannot be claimed as
a differentiator. The doc asks which of five layers really differ once the
baseline is given the same ledger machinery:

  1 generic security event logging
  2 hash-chain tamper detection
  3 replicated-ledger recovery of evidence
  4 clause -> control id -> evidence -> severity -> enforcement -> reputation
  5 reproducibility of past verdicts after a policy change

Layers 1-3 are properties of the ledger, not of the control taxonomy; give the
baseline the same append-only hash chain and they are identical by
construction, which this script confirms rather than assumes. Layer 4 is a
design property and is not measurable as a rate. Layer 5 is measurable and is
what this experiment isolates.

The measurement. Two ledgers are written from the SAME run. The governance
ledger records the per-control evidence of all 36 controls. The conventional
ledger records what a conventional pipeline consumes: timestamps, payload
hashes, signatures and the update magnitude -- it never records a label
distribution, because it runs no control that reads one. A policy is then
changed, and for every (agent, round) we ask whether the verdict under the NEW
policy can be recomputed from the stored evidence alone, without re-running any
training.

Policy changes tested
  PC1 adopt S-DF-03   a control the conventional pipeline was not running
  PC2 statistical findings reclassified from Reject to Review severity
  PC3 retire M-SC-02

Usage
  python e6_audit.py run <cell> <seed> <out.jsonl>
  python e6_audit.py agg <dir> <out.json>
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[2]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                              # noqa: E402
from trust import fl_simulation as fl                                 # noqa: E402
from trust.engine import build_ctx                                     # noqa: E402
from trust.ledger import Ledger, sha256_hex, sign                     # noqa: E402
from trust.taxonomy import ALL_CONTROLS                               # noqa: E402
from trust.engine import FAIL_SUBSCORE                                 # noqa: E402

ALL_IDS = {cp.id for cp in ALL_CONTROLS}
SEVERITY = {cp.id: cp.severity for cp in ALL_CONTROLS}
STATISTICAL = {"M-BC-01", "M-BC-02", "M-SC-02", "S-DF-03"}
CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02", "M-BC-01"}

# What each ledger variant retains in a record's meta. Block-level fields
# (timestamp, ingest_time, payload hash, signature, chain hashes) are always
# stored by the ledger itself and are therefore available to both.
GOV_META = ("round", "behaviour", "local_acc", "update_norm", "label_kl",
            "logged", "model_version")
CONV_META = ("round", "behaviour", "update_norm", "logged", "model_version")

ATTACK_START = 5
RETURN_ROUND = 15
RETURNING = {5, 6}
REVOKED = 7
BASE_PROFILE = {1: "label_flip", 3: "scale_attack", 5: "stale", 6: "tamper"}

EVAL_SEEDS = list(range(400, 420))      # 20 seeds, disjoint from every other run
CELLS = {
    "r25_iid": dict(rounds=25, partition="iid", alpha=1.0, flip=0.70),
    "r25_dir": dict(rounds=25, partition="dirichlet", alpha=0.5, flip=0.70),
}


def evaluate_safe(ctx):
    """Evaluate every control, recording the ones the stored evidence cannot
    support.

    A conventional pipeline records only what its own checks consume, so a
    control that reads something it never kept -- a label distribution, for
    instance -- cannot be computed at all. That is the outcome this experiment
    measures, so it is captured rather than allowed to abort the run.
    """
    from trust.engine import _MODULES, _resolve  # noqa: F401
    from trust.attestations import evaluate_attestation
    results, missing = {}, []
    for cp in ALL_CONTROLS:
        try:
            if cp.method == "attestation" or not cp.check:
                results[cp.id] = evaluate_attestation(cp.id)
            else:
                results[cp.id] = _resolve(cp.check)(ctx)
        except Exception:
            missing.append(cp.id)
    return results, missing


def verdict_from(results, missing, aid, disabled, severity_map):
    """The severity-driven verdict, with the policy supplied explicitly.

    This is the framework's own rule with the policy as a parameter, which is
    exactly what "replay a past decision under a changed policy" requires. If
    the active policy depends on a control the stored evidence cannot support,
    the decision is not reproducible and that is reported as such rather than
    silently scored as agreement.
    """
    if any(c not in disabled for c in missing):
        return "UNRECONSTRUCTIBLE"
    failed = [cp.id for cp in ALL_CONTROLS
              if cp.id not in disabled
              and cp.id in results
              and aid in results[cp.id].per_agent
              and results[cp.id].per_agent[aid] < FAIL_SUBSCORE]
    sev = {severity_map.get(cid, SEVERITY[cid]) for cid in failed}
    if "reject" in sev:
        return "Untrusted"
    if "review" in sev:
        return "Under Review"
    return "Trusted"


def behaviour_at(aid, rnd):
    if rnd < ATTACK_START:
        return "honest"
    if aid in RETURNING and rnd >= RETURN_ROUND:
        return "honest"
    return BASE_PROFILE.get(aid, "honest")


def run(cell_key, seed):
    cell = CELLS[cell_key]
    rounds, flip = cell["rounds"], cell["flip"]
    prep = fl.prepare(seed, partition=cell["partition"], alpha=cell["alpha"])
    if "554" not in prep["dataset_name"] or prep["X_tr"].shape[1] != 784:
        raise SystemExit("wrong dataset: %s" % prep["dataset_name"])

    rng = np.random.RandomState(seed)
    backbone = prep.get("backbone", "linear")
    X_tr, y_tr = prep["X_tr"], prep["y_tr"]
    X_te, y_te = prep["X_te"], prep["y_te"]
    shards, base_dist = prep["shards"], prep["base_dist"]
    classes = np.unique(y_tr)
    ncls = prep["n_classes"]

    def local_fit(gW, Xl, yl):
        m = fl._new_model(backbone, seed)
        m.partial_fit(Xl[:256], yl[:256], classes=classes)
        fl._set_w(m, gW[0], gW[1])
        for _ in range(fl.EPOCHS):
            order = rng.permutation(len(Xl))
            m.partial_fit(Xl[order], yl[order])
        return m

    def wavg(Ws, weights):
        w = np.asarray(weights, float)
        w /= w.sum()
        n_c, n_i = len(Ws[0][0]), len(Ws[0][1])
        return ([sum(w[k] * Ws[k][0][l] for k in range(len(Ws)))
                 for l in range(n_c)],
                [sum(w[k] * Ws[k][1][l] for k in range(len(Ws)))
                 for l in range(n_i)])

    ledger = Ledger()
    ledger.reset()
    gm = fl._new_model(backbone, seed)
    gm.partial_fit(X_tr[:256], y_tr[:256], classes=classes)

    agents = {sh["agent_id"]: {
        "agent_id": sh["agent_id"], "principal": fl._principal(sh["agent_id"]),
        "behaviour": "honest", "is_adversary": False,
        "n_samples": int(len(sh["y"])), "rounds": []} for sh in shards}

    for rnd in range(rounds):
        now = 1_700_000_000 + rnd * 3600
        gW = fl._get_w(gm)
        submitted = []
        for sh in shards:
            aid = sh["agent_id"]
            beh = behaviour_at(aid, rnd)
            Xl, yl = sh["X"].copy(), sh["y"].copy()
            if beh == "label_flip":
                yl = yl.copy()
                yl[rng.rand(len(yl)) < flip] = 0
            m = local_fit(gW, Xl, yl)
            lW = fl._get_w(m)
            d_c = [a - b for a, b in zip(lW[0], gW[0])]
            d_i = [a - b for a, b in zip(lW[1], gW[1])]
            if beh == "scale_attack":
                d_c = [x * 12.0 for x in d_c]
                d_i = [x * 12.0 for x in d_i]
            new_c = [g + d for g, d in zip(gW[0], d_c)]
            new_i = [g + d for g, d in zip(gW[1], d_i)]
            update_norm = float(sum(np.linalg.norm(a) for a in d_c)
                                + sum(np.linalg.norm(a) for a in d_i))
            local_acc = float(m.score(X_te, y_te))
            klv = fl._label_kl(yl, ncls, base_dist)
            payload = ("agent%d-round%d-n%d" % (aid, rnd, len(yl))).encode()
            claimed_hash = signature = None
            ts = now
            if beh == "stale":
                ts = now - config.FRESHNESS_SLA_SEC * 5
            if beh == "tamper":
                claimed_hash = sha256_hex(payload + b"-tampered")
                signature = sign(aid, sha256_hex(payload))
            blk = ledger.append(
                agent_id=aid, principal=fl._principal(aid),
                tx_type="data_packet", payload=payload,
                claimed_hash=claimed_hash, signature=signature,
                timestamp=ts, ingest_time=now,
                meta={"round": rnd, "behaviour": beh,
                      "local_acc": round(local_acc, 4),
                      "update_norm": round(update_norm, 6),
                      "label_kl": round(klv, 6), "logged": True,
                      "model_version": rnd + 1})
            agents[aid]["rounds"].append({
                "round": rnd, "update_norm": update_norm,
                "local_acc": local_acc, "label_kl": klv,
                "block_index": blk.index, "timestamp": ts,
                "ingest_time": now, "model_version": rnd + 1})
            agents[aid]["behaviour"] = beh
            submitted.append({"W": (new_c, new_i), "weight": len(yl)})
        cc, ii = wavg([c["W"] for c in submitted],
                      [c["weight"] for c in submitted])
        fl._set_w(gm, cc, ii)

    now = 1_700_000_000 + (rounds - 1) * 3600

    def signals_from_ledger(keep_keys):
        """Rebuild the adjudication input from stored records only.

        A field a ledger variant never recorded is simply absent, which is the
        situation an auditor faces when a policy starts to depend on evidence
        the pipeline was not keeping.
        """
        blocks = Ledger().read()
        reb = {}
        for b in blocks:
            aid = int(b["agent_id"])
            meta = {k: v for k, v in (b.get("meta") or {}).items()
                    if k in keep_keys}
            a = reb.setdefault(aid, {
                "agent_id": aid, "principal": b.get("principal"),
                "behaviour": "honest", "is_adversary": False,
                "n_samples": 0, "rounds": []})
            row = {"round": meta.get("round"),
                   "block_index": b.get("index"),
                   "timestamp": b.get("timestamp"),
                   "ingest_time": b.get("ingest_time"),
                   "model_version": meta.get("model_version")}
            if "update_norm" in meta:
                row["update_norm"] = meta["update_norm"]
            if "label_kl" in meta:
                row["label_kl"] = meta["label_kl"]
            if "local_acc" in meta:
                row["local_acc"] = meta["local_acc"]
            a["rounds"].append(row)
            a["behaviour"] = meta.get("behaviour", "honest")
        for aid, a in reb.items():
            a["n_samples"] = agents[aid]["n_samples"]
        return {
            "dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
            "num_rounds": rounds, "global_accuracy": 0.0,
            "revoked_agent": REVOKED,
            "signing_scheme": config.SIGNING_SCHEME,
            "agents": reb, "seed": seed, "generated_at": now}

    live_signals = {
        "dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
        "num_rounds": rounds, "global_accuracy": 0.0,
        "revoked_agent": REVOKED, "signing_scheme": config.SIGNING_SCHEME,
        "agents": agents, "seed": seed, "generated_at": now}

    res_live, miss_live = evaluate_safe(build_ctx(live_signals))
    res_gov, miss_gov = evaluate_safe(build_ctx(signals_from_ledger(GOV_META)))
    res_conv, miss_conv = evaluate_safe(build_ctx(signals_from_ledger(CONV_META)))
    aids = sorted(agents)

    # The policy each pipeline was operating under when the decisions were
    # taken. The conventional pipeline ran only its own control set, so
    # reproducing ITS OWN verdicts never needed label-distribution evidence --
    # which is why the online-adjudication table reports 1.000 for both.
    LEDGERS = {
        "governance_ledger": dict(res=res_gov, missing=miss_gov,
                                  base_disabled=frozenset()),
        "conventional_ledger": dict(res=res_conv, missing=miss_conv,
                                    base_disabled=frozenset(ALL_IDS - CONV_NORM)),
    }

    # Policy changes an operator could plausibly make. Each is expressed
    # relative to the pipeline's own base policy.
    POLICY_CHANGES = {
        # the deployment adopts the label-distribution control it was not using
        "PC1_adopt_sdf03": dict(enable={"S-DF-03"}, disable=set(),
                                severity={"S-DF-03": "review"}),
        # statistical findings reclassified from Reject to Review
        "PC2_stat_to_review": dict(enable=set(), disable=set(),
                                   severity={c: "review" for c in STATISTICAL}),
        # a control is retired
        "PC3_retire_msc02": dict(enable=set(), disable={"M-SC-02"}, severity={}),
    }

    out = {"cell": cell_key, "seed": seed, "rounds": rounds,
           "n_agents": len(aids), "replay": {},
           "controls_not_evaluable": {"governance_ledger": sorted(miss_gov),
                                      "conventional_ledger": sorted(miss_conv),
                                      "live": sorted(miss_live)}}

    # Under each pipeline's own policy: the reconstruction rate reported in
    # the online-adjudication table.
    for name, L in LEDGERS.items():
        dis = L["base_disabled"]
        ok = 0
        for a in aids:
            v_stored = verdict_from(L["res"], L["missing"], a, dis, {})
            v_live = verdict_from(res_live, miss_live, a, dis, {})
            ok += int(v_stored == v_live
                      and v_stored != "UNRECONSTRUCTIBLE")
        out.setdefault("replay_same_policy", {})[name] = {
            "matched": ok, "total": len(aids),
            "rate": round(ok / len(aids), 4)}

    # After a policy change.
    for pc, spec in POLICY_CHANGES.items():
        row = {}
        for name, L in LEDGERS.items():
            dis = (set(L["base_disabled"]) - spec["enable"]) | spec["disable"]
            dis = frozenset(dis)
            ok = 0
            for a in aids:
                v_stored = verdict_from(L["res"], L["missing"], a, dis,
                                        spec["severity"])
                v_live = verdict_from(res_live, miss_live, a, dis,
                                      spec["severity"])
                ok += int(v_stored == v_live
                          and v_stored != "UNRECONSTRUCTIBLE")
            blocked = sorted(c for c in L["missing"] if c not in dis)
            row[name] = {"matched": ok, "total": len(aids),
                         "rate": round(ok / len(aids), 4),
                         "blocked_by_missing_evidence": blocked}
        out["replay"][pc] = row

    # Layers 1-3 are properties of the ledger, not of the control taxonomy, so
    # measure them rather than assert them: verify the chain, then rewrite one
    # stored record and verify again.
    lg = Ledger()
    before = lg.verify_chain()
    path = lg.path
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    mid = len(lines) // 2
    rec = json.loads(lines[mid])
    rec["meta"]["update_norm"] = 0.0
    lines[mid] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    after = Ledger().verify_chain()
    path.write_text(original, encoding="utf-8")
    restored = Ledger().verify_chain()
    out["ledger_layers"] = {
        "events_logged": before["blocks"],
        "continuity_before": before["continuity"],
        "block_hash_integrity_before": before["block_hash_integrity"],
        "block_hash_integrity_after_edit": after["block_hash_integrity"],
        "tamper_detected": after["block_hash_integrity"] < 1.0,
        "restored_ok": restored["block_hash_integrity"] == 1.0,
        "chain_consistent": before["continuity"] == 1.0,
        "note": ("this ledger machinery is identical for both variants, so "
                 "layers 1-3 cannot distinguish them; only the retained "
                 "evidence does"),
    }
    out["evidence_fields"] = {"governance_ledger": list(GOV_META),
                              "conventional_ledger": list(CONV_META)}
    return out


def mode_run(cell, seed, out_path):
    r = run(cell, int(seed))
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n")
    return 0


def mode_agg(d, out_path):
    rows = []
    for p in sorted(Path(d).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no records in %s" % d)
    summary = {}
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell]
        if not rs:
            continue
        s = {"n_seeds": len(rs)}
        for name in ("governance_ledger", "conventional_ledger"):
            s.setdefault("same_policy", {})[name] = round(st.mean(
                r["replay_same_policy"][name]["rate"] for r in rs), 4)
        for pc in rs[0]["replay"]:
            for name in ("governance_ledger", "conventional_ledger"):
                s.setdefault(pc, {})[name] = round(st.mean(
                    r["replay"][pc][name]["rate"] for r in rs), 4)
        s["chain_consistent_all_seeds"] = all(
            r["ledger_layers"]["chain_consistent"] for r in rs)
        s["events_logged_mean"] = round(st.mean(
            r["ledger_layers"]["events_logged"] for r in rs), 1)
        summary[cell] = s
    Path(out_path).write_text(json.dumps({
        "experiment": "E6 / P0-2 auditability layers",
        "evaluation_seeds": EVAL_SEEDS, "cells": CELLS,
        "evidence_fields": rows[0]["evidence_fields"],
        "n_records": len(rows), "summary": summary,
    }, indent=1), encoding="utf-8")
    print("wrote", out_path, "from", len(rows), "records")
    for cell, s in summary.items():
        print("\n==", cell, " seeds", s["n_seeds"])
        for k, v in s.items():
            if isinstance(v, dict):
                print("   %-20s gov %.4f   conv %.4f"
                      % (k, v["governance_ledger"], v["conventional_ledger"]))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        raise SystemExit(mode_run(a[1], a[2], a[3]))
    if a[0] == "agg":
        raise SystemExit(mode_agg(a[1], a[2]))
    raise SystemExit("unknown mode %r" % a[0])
