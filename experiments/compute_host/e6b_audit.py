# -*- coding: utf-8 -*-
"""E6b: is the replay advantage the taxonomy, or just a decision to log more?

E6 compared a governance record against a conventional one and found they
differ on exactly one scenario: adopting a control the pipeline was not
running. But the governance record stored a label divergence computed from the
true local labels, which a deployed server does not have, and the conventional
record stored no label evidence at all. Two objections follow, and this
experiment answers both.

  1 the governance record's advantage may rest on privileged evidence
  2 a conventional pipeline that simply chose to log more might match it

Four records are written from the same run:

  governance_privileged      per-control evidence of all 36 controls, with the
                             label divergence from the true labels
                             (REFERENCE -- not deployable)
  governance_deployable      the same, with the divergence from the
                             update-derived estimate a server can compute
  conventional_minimal       only what conventional checks consume: timestamps,
                             payload hashes, authentication tags, update norm
  conventional_full_logging  conventional checks, but the pipeline also logs
                             the update-derived estimate although it runs no
                             control that reads one

If conventional_full_logging also replays the policy change, then the property
belongs to the decision about what evidence to retain, not to the control
taxonomy, and the paper must say so.

Usage
  python e6b_audit.py run <cell> <seed> <thresholds.json> <out.jsonl>
  python e6b_audit.py agg <dir> <out.json>
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
from trust.engine import build_ctx, FAIL_SUBSCORE, _resolve           # noqa: E402
from trust.attestations import evaluate_attestation                   # noqa: E402
from trust.ledger import Ledger, sha256_hex, sign                     # noqa: E402
from trust.taxonomy import ALL_CONTROLS                               # noqa: E402

EPS = 1e-12
SDF03_FAIL_KL = 0.60
ALL_IDS = {cp.id for cp in ALL_CONTROLS}
SEVERITY = {cp.id: cp.severity for cp in ALL_CONTROLS}
STATISTICAL = {"M-BC-01", "M-BC-02", "M-SC-02", "S-DF-03"}
CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02", "M-BC-01"}

ATTACK_START = 5
RETURN_ROUND = 15
RETURNING = {5, 6}
REVOKED = 7
BASE_PROFILE = {1: "label_flip", 3: "scale_attack", 5: "stale", 6: "tamper"}

EVAL_SEEDS = list(range(600, 620))     # 20 seeds, disjoint from every other run
CELLS = {
    "r25_iid": dict(rounds=25, partition="iid", alpha=1.0, flip=0.70,
                    thr_cell="r25_iid_f70"),
    "r25_dir": dict(rounds=25, partition="dirichlet", alpha=0.5, flip=0.70,
                    thr_cell="r25_dir_f70"),
}

# What each record keeps, and which stored divergence the control reads.
VARIANTS = {
    "governance_privileged": dict(
        keys=("round", "behaviour", "local_acc", "update_norm", "logged",
              "model_version", "label_kl_true"),
        kl="label_kl_true", base_disabled=frozenset(), deployable=False),
    "governance_deployable": dict(
        keys=("round", "behaviour", "local_acc", "update_norm", "logged",
              "model_version", "label_kl_est"),
        kl="label_kl_est", base_disabled=frozenset(), deployable=True),
    "conventional_minimal": dict(
        keys=("round", "behaviour", "update_norm", "logged", "model_version"),
        kl=None, base_disabled=frozenset(ALL_IDS - CONV_NORM), deployable=True),
    "conventional_full_logging": dict(
        keys=("round", "behaviour", "update_norm", "logged", "model_version",
              "label_kl_est"),
        kl="label_kl_est", base_disabled=frozenset(ALL_IDS - CONV_NORM),
        deployable=True),
}

POLICY_CHANGES = {
    "PC1_adopt_sdf03": dict(enable={"S-DF-03"}, disable=set(),
                            severity={"S-DF-03": "review"}),
    "PC2_stat_to_review": dict(enable=set(), disable=set(),
                               severity={c: "review" for c in STATISTICAL}),
    "PC3_retire_msc02": dict(enable=set(), disable={"M-SC-02"}, severity={}),
}


def kl(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return float(np.sum(np.where(p > 0, p * np.log((p + EPS) / (q + EPS)), 0.0)))


def estimate_dist(delta_b, prior):
    d = np.asarray(delta_b, float) - np.mean(delta_b)
    if np.allclose(d, 0):
        return np.asarray(prior, float)
    neg = d < 0
    kappa = float(np.min(-np.asarray(prior)[neg] / d[neg])) if neg.any() else 1.0
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 1.0
    est = np.clip(np.asarray(prior, float) + kappa * d, 0.0, None)
    t = est.sum()
    return est / t if t > 0 else np.asarray(prior, float)


def flat(W):
    return np.concatenate([np.asarray(x, float).ravel() for x in W[0]]
                          + [np.asarray(y, float).ravel() for y in W[1]])


def evaluate_safe(ctx):
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
    if any(c not in disabled for c in missing):
        return "UNRECONSTRUCTIBLE"
    failed = [cp.id for cp in ALL_CONTROLS
              if cp.id not in disabled and cp.id in results
              and aid in results[cp.id].per_agent
              and results[cp.id].per_agent[aid] < FAIL_SUBSCORE]
    sev = {severity_map.get(c, SEVERITY[c]) for c in failed}
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


def run(cell_key, seed, thr_path):
    cell = CELLS[cell_key]
    rounds, flip = cell["rounds"], cell["flip"]
    thr = json.loads(Path(thr_path).read_text(encoding="utf-8"))
    tau = float(thr["thresholds"]["gradient"][cell["thr_cell"]])

    prep = fl.prepare(seed, partition=cell["partition"], alpha=cell["alpha"])
    if "554" not in prep["dataset_name"] or prep["X_tr"].shape[1] != 784:
        raise SystemExit("wrong dataset: %s" % prep["dataset_name"])

    rng = np.random.RandomState(seed)
    X_tr, y_tr = prep["X_tr"], prep["y_tr"]
    X_te, y_te = prep["X_te"], prep["y_te"]
    shards, base_dist = prep["shards"], prep["base_dist"]
    classes = np.unique(y_tr)
    ncls = prep["n_classes"]
    backbone = prep.get("backbone", "linear")

    def local_fit(gW, Xl, yl):
        m = fl._new_model(backbone, seed)
        m.partial_fit(Xl[:256], yl[:256], classes=classes)
        fl._set_w(m, gW[0], gW[1])
        for _ in range(fl.EPOCHS):
            order = rng.permutation(len(Xl))
            m.partial_fit(Xl[order], yl[order])
        return m

    def wavg(Ws, ws):
        w = np.asarray(ws, float)
        w /= w.sum()
        return ([sum(w[k] * Ws[k][0][l] for k in range(len(Ws)))
                 for l in range(len(Ws[0][0]))],
                [sum(w[k] * Ws[k][1][l] for k in range(len(Ws)))
                 for l in range(len(Ws[0][1]))])

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
        sub = []
        for sh in shards:
            aid = sh["agent_id"]
            beh = behaviour_at(aid, rnd)
            Xl, yl = sh["X"].copy(), sh["y"].copy()
            if beh == "label_flip":
                yl = yl.copy()
                yl[rng.rand(len(yl)) < flip] = 0
            m = local_fit(gW, Xl, yl)
            lW = fl._get_w(m)
            d_c = [p - q for p, q in zip(lW[0], gW[0])]
            d_i = [p - q for p, q in zip(lW[1], gW[1])]
            if beh == "scale_attack":
                d_c = [x * 12.0 for x in d_c]
                d_i = [x * 12.0 for x in d_i]
            new_c = [g + d for g, d in zip(gW[0], d_c)]
            new_i = [g + d for g, d in zip(gW[1], d_i)]
            dflat = flat((d_c, d_i))
            un = float(sum(np.linalg.norm(x) for x in d_c)
                       + sum(np.linalg.norm(y) for y in d_i))
            kl_true = fl._label_kl(yl, ncls, base_dist)
            kl_est = SDF03_FAIL_KL * kl(
                estimate_dist(dflat[-ncls:], base_dist), base_dist) / max(tau, EPS)
            payload = ("agent%d-round%d-n%d" % (aid, rnd, len(yl))).encode()
            ch = sg = None
            ts = now
            if beh == "stale":
                ts = now - config.FRESHNESS_SLA_SEC * 5
            if beh == "tamper":
                ch = sha256_hex(payload + b"-tampered")
                sg = sign(aid, sha256_hex(payload))
            blk = ledger.append(
                agent_id=aid, principal=fl._principal(aid),
                tx_type="data_packet", payload=payload,
                claimed_hash=ch, signature=sg, timestamp=ts, ingest_time=now,
                meta={"round": rnd, "behaviour": beh,
                      "local_acc": 0.0, "update_norm": round(un, 6),
                      "label_kl_true": round(kl_true, 6),
                      "label_kl_est": round(kl_est, 6),
                      "label_kl": round(kl_true, 6),
                      "logged": True, "model_version": rnd + 1})
            agents[aid]["rounds"].append({
                "round": rnd, "update_norm": un, "local_acc": 0.0,
                "label_kl": kl_true, "block_index": blk.index,
                "timestamp": ts, "ingest_time": now,
                "model_version": rnd + 1})
            agents[aid]["behaviour"] = beh
            sub.append({"W": (new_c, new_i), "weight": len(yl)})
        cc, ii = wavg([c["W"] for c in sub], [c["weight"] for c in sub])
        fl._set_w(gm, cc, ii)

    now = 1_700_000_000 + (rounds - 1) * 3600

    def signals_from(keys, kl_key):
        blocks = Ledger().read()
        reb = {}
        for b in blocks:
            aid = int(b["agent_id"])
            meta = {k: v for k, v in (b.get("meta") or {}).items() if k in keys}
            a = reb.setdefault(aid, {
                "agent_id": aid, "principal": b.get("principal"),
                "behaviour": "honest", "is_adversary": False,
                "n_samples": agents[aid]["n_samples"], "rounds": []})
            row = {"round": meta.get("round"), "block_index": b.get("index"),
                   "timestamp": b.get("timestamp"),
                   "ingest_time": b.get("ingest_time"),
                   "model_version": meta.get("model_version")}
            if "update_norm" in meta:
                row["update_norm"] = meta["update_norm"]
            if "local_acc" in meta:
                row["local_acc"] = meta["local_acc"]
            if kl_key and kl_key in meta:
                row["label_kl"] = meta[kl_key]
            a["rounds"].append(row)
            a["behaviour"] = meta.get("behaviour", "honest")
        return {"dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
                "num_rounds": rounds, "global_accuracy": 0.0,
                "revoked_agent": REVOKED,
                "signing_scheme": config.SIGNING_SCHEME,
                "agents": reb, "seed": seed, "generated_at": now}

    live = {"dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
            "num_rounds": rounds, "global_accuracy": 0.0,
            "revoked_agent": REVOKED, "signing_scheme": config.SIGNING_SCHEME,
            "agents": agents, "seed": seed, "generated_at": now}
    res_live, miss_live = evaluate_safe(build_ctx(live))
    aids = sorted(agents)

    built = {}
    for name, v in VARIANTS.items():
        r, m = evaluate_safe(build_ctx(signals_from(v["keys"], v["kl"])))
        built[name] = (r, m)

    out = {"cell": cell_key, "seed": seed, "rounds": rounds,
           "gradient_tau": tau, "n_agents": len(aids),
           "same_policy": {}, "replay": {},
           "controls_not_evaluable": {n: sorted(built[n][1]) for n in VARIANTS}}

    for name, v in VARIANTS.items():
        r, m = built[name]
        dis = v["base_disabled"]
        ok = sum(1 for a in aids
                 if verdict_from(r, m, a, dis, {})
                 == verdict_from(res_live, miss_live, a, dis, {})
                 and verdict_from(r, m, a, dis, {}) != "UNRECONSTRUCTIBLE")
        out["same_policy"][name] = round(ok / len(aids), 4)

    for pc, spec in POLICY_CHANGES.items():
        row = {}
        for name, v in VARIANTS.items():
            r, m = built[name]
            dis = frozenset((set(v["base_disabled"]) - spec["enable"])
                            | spec["disable"])
            ok = sum(1 for a in aids
                     if verdict_from(r, m, a, dis, spec["severity"])
                     == verdict_from(res_live, miss_live, a, dis,
                                     spec["severity"])
                     and verdict_from(r, m, a, dis, spec["severity"])
                     != "UNRECONSTRUCTIBLE")
            row[name] = round(ok / len(aids), 4)
        out["replay"][pc] = row

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
    out["ledger"] = {
        "events": before["blocks"],
        "continuity": before["continuity"],
        "hash_integrity_before": before["block_hash_integrity"],
        "hash_integrity_after_edit": after["block_hash_integrity"],
        "tamper_detected": after["block_hash_integrity"] < 1.0,
    }
    return out


def mode_run(cell, seed, thr, out_path):
    r = run(cell, int(seed), thr)
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
        s = {"n_seeds": len(rs), "gradient_tau": rs[0]["gradient_tau"],
             "same_policy": {}, "replay": {}}
        for n in VARIANTS:
            s["same_policy"][n] = round(st.mean(r["same_policy"][n]
                                                for r in rs), 4)
        for pc in rs[0]["replay"]:
            s["replay"][pc] = {n: round(st.mean(r["replay"][pc][n] for r in rs), 4)
                               for n in VARIANTS}
        s["tamper_detected_all_seeds"] = all(r["ledger"]["tamper_detected"]
                                             for r in rs)
        s["continuity_all_seeds"] = all(r["ledger"]["continuity"] == 1.0
                                        for r in rs)
        summary[cell] = s
    Path(out_path).write_text(json.dumps({
        "experiment": "E6b / audit replay with evidence provenance separated",
        "evaluation_seeds": EVAL_SEEDS, "cells": CELLS,
        "variants": {k: {"keys": list(v["keys"]), "kl_source": v["kl"],
                         "deployable": v["deployable"]}
                     for k, v in VARIANTS.items()},
        "n_records": len(rows), "summary": summary}, indent=1),
        encoding="utf-8")
    print("wrote", out_path, "from", len(rows), "records")
    for cell, s in summary.items():
        print("\n== %s  seeds %d  gradient tau %.4f" % (cell, s["n_seeds"],
                                                        s["gradient_tau"]))
        hdr = "%-26s %10s %10s %10s %10s" % (
            "scenario", "gov-priv", "gov-depl", "conv-min", "conv-full")
        print(hdr)
        print("%-26s %10.4f %10.4f %10.4f %10.4f" % (
            "same policy",
            s["same_policy"]["governance_privileged"],
            s["same_policy"]["governance_deployable"],
            s["same_policy"]["conventional_minimal"],
            s["same_policy"]["conventional_full_logging"]))
        for pc, v in s["replay"].items():
            print("%-26s %10.4f %10.4f %10.4f %10.4f" % (
                pc, v["governance_privileged"], v["governance_deployable"],
                v["conventional_minimal"], v["conventional_full_logging"]))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        raise SystemExit(mode_run(a[1], a[2], a[3], a[4]))
    if a[0] == "agg":
        raise SystemExit(mode_agg(a[1], a[2]))
    raise SystemExit("unknown mode %r" % a[0])
