# -*- coding: utf-8 -*-
"""E1: online adjudication with persistent enforcement.

The round loop of `train_trajectory` runs to completion before any control is
evaluated, so it cannot express a server that adjudicates as it goes. This
driver reimplements that loop with the round boundary under our control, so
every verdict uses only the evidence written to the ledger up to that round.

Local training, behaviour injection, ledger records and signal fields are
transcribed from `fl_simulation.train_trajectory` so the two remain
comparable.

Schedule: all participants behave honestly until ATTACK_START; the five
threat roles then begin; two of them (replay and tampering) return to honest
behaviour at RETURN_ROUND, which tests whether an excluded participant can be
re-admitted.

Conditions
  A  conventional security checks + norm filter, adjudicated independently
     every round, no memory between rounds
  B  the same control set, with persistent exclusion
  C  full governance control set, no persistent exclusion
  D  full governance, persistent exclusion, ledger retained   (proposed)
  E  Multi-Krum alone, no adjudication                        (reference)

Usage: python e1_online.py <out.json> [rounds] [seed_start] [seed_count]
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                        # noqa: E402
from trust import aggregators as agg                             # noqa: E402
from trust import fl_simulation as fl                            # noqa: E402
from trust.ledger import sha256_hex, sign                        # noqa: E402
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402
from trust.ledger import Ledger                                  # noqa: E402
from trust.taxonomy import ALL_CONTROLS                          # noqa: E402

ALL_IDS = {cp.id for cp in ALL_CONTROLS}
CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02", "M-BC-01"}

ATTACK_START = 5
RETURN_ROUND = 15
RETURNING = {5, 6}          # replay and tampering stop misbehaving
REVOKED = 7

BASE_PROFILE = {1: "label_flip", 3: "scale_attack", 5: "stale", 6: "tamper"}

CONDITIONS = {
    "A_conv_perround":   dict(controls=CONV_NORM, persist=False, aggr=None),
    "B_conv_persist":    dict(controls=CONV_NORM, persist=True,  aggr=None),
    "C_gov_perround":    dict(controls=ALL_IDS,   persist=False, aggr=None),
    "D_gov_persist":     dict(controls=ALL_IDS,   persist=True,  aggr=None),
    "E_multikrum":       dict(controls=None,      persist=False,
                              aggr=agg.multi_krum),
}


def behaviour_at(aid, rnd):
    """What the participant actually does in this round."""
    if rnd < ATTACK_START:
        return "honest"
    if aid in RETURNING and rnd >= RETURN_ROUND:
        return "honest"
    return BASE_PROFILE.get(aid, "honest")


def adversarial_at(aid, rnd):
    """Is this participant in violation in this round (ground truth)?"""
    if rnd < ATTACK_START:
        return False
    if aid == REVOKED:
        return True                      # revoked from ATTACK_START onwards
    if aid in RETURNING and rnd >= RETURN_ROUND:
        return False
    return aid in BASE_PROFILE


def run_condition(prep, seed, cond, rounds):
    cfg = CONDITIONS[cond]
    rng = np.random.RandomState(seed)
    backbone = prep.get("backbone", "linear")
    X_tr, y_tr = prep["X_tr"], prep["y_tr"]
    X_te, y_te = prep["X_te"], prep["y_te"]
    shards, base_dist = prep["shards"], prep["base_dist"]
    classes = np.unique(y_tr)

    def new_model():
        m = fl._new_model(backbone, seed)
        m.partial_fit(X_tr[:256], y_tr[:256], classes=classes)
        return m

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
        coefs = [sum(w[k] * Ws[k][0][l] for k in range(len(Ws)))
                 for l in range(n_c)]
        ints = [sum(w[k] * Ws[k][1][l] for k in range(len(Ws)))
                for l in range(n_i)]
        return coefs, ints

    ledger = Ledger()
    ledger.reset()
    gm = new_model()

    agents = {sh["agent_id"]: {
        "agent_id": sh["agent_id"], "principal": fl._principal(sh["agent_id"]),
        "behaviour": "honest", "is_adversary": False,
        "n_samples": int(len(sh["y"])), "rounds": []} for sh in shards}

    excluded = set()                       # persistent, carried across rounds
    adv_admitted = 0                       # cumulative adversarial updates aggregated
    first_detect, first_isolate = {}, {}
    honest_wrong_isolation = {}            # aid -> rounds spent wrongly excluded
    per_round = []
    recon_ok = recon_total = 0

    for rnd in range(rounds):
        now = 1_700_000_000 + rnd * 3600
        gW = fl._get_w(gm)
        submitted = []

        for sh in shards:
            aid = sh["agent_id"]
            if aid in excluded:
                continue                   # an excluded participant stops submitting
            beh = behaviour_at(aid, rnd)
            Xl, yl = sh["X"].copy(), sh["y"].copy()
            if beh in ("label_flip", "collude_flip"):
                yl = yl.copy()
                yl[rng.rand(len(yl)) < 0.7] = 0

            m = local_fit(gW, Xl, yl)
            lW = fl._get_w(m)
            d_c = [a - b for a, b in zip(lW[0], gW[0])]
            d_i = [a - b for a, b in zip(lW[1], gW[1])]
            if beh == "scale_attack":
                d_c = [x * 12.0 for x in d_c]; d_i = [x * 12.0 for x in d_i]
            elif beh == "stealth_scale":
                f = config.STEALTH_SCALE_FACTOR
                d_c = [x * f for x in d_c]; d_i = [x * f for x in d_i]

            new_c = [g + d for g, d in zip(gW[0], d_c)]
            new_i = [g + d for g, d in zip(gW[1], d_i)]

            update_norm = float(sum(np.linalg.norm(a) for a in d_c)
                                + sum(np.linalg.norm(a) for a in d_i))
            local_acc = float(m.score(X_te, y_te))
            klv = fl._label_kl(yl, prep["n_classes"], base_dist)
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
            agents[aid]["is_adversary"] = adversarial_at(aid, rnd)
            submitted.append({"W": (new_c, new_i), "weight": len(yl),
                              "aid": aid, "coef": new_c[0], "int": new_i[0]})

        # ---- adjudication on the evidence written so far -------------------
        flagged = set()
        if cfg["controls"] is not None:
            signals = {
                "dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
                "num_rounds": rnd + 1, "global_accuracy": 0.0,
                "revoked_agent": REVOKED if rnd >= ATTACK_START else None,
                "signing_scheme": config.SIGNING_SCHEME,
                "agents": {a: v for a, v in agents.items() if v["rounds"]},
                "seed": seed, "generated_at": now}
            ctx = build_ctx(signals)
            res = evaluate_controls(ctx)
            disabled = frozenset(ALL_IDS - cfg["controls"])
            for a in ctx.agent_ids():
                if a in excluded:
                    continue
                if agent_verdict(res, a, disabled=disabled)[0] != "Trusted":
                    flagged.add(a)

            # audit reconstruction: rebuild the same decision from the ledger
            blocks = Ledger().read()
            rebuilt = {}
            for b in blocks:
                aid = int(b["agent_id"]); meta = b.get("meta", {})
                rebuilt.setdefault(aid, []).append(meta)
            for a in ctx.agent_ids():
                recon_total += 1
                if a in rebuilt and len(rebuilt[a]) == len(agents[a]["rounds"]):
                    recon_ok += 1

        for a in flagged:
            first_detect.setdefault(a, rnd)
        admitted = [c for c in submitted if c["aid"] not in flagged]
        for c in submitted:
            if c["aid"] in flagged:
                first_isolate.setdefault(c["aid"], rnd)
        if cfg["persist"]:
            excluded |= flagged

        for c in admitted:
            if adversarial_at(c["aid"], rnd):
                adv_admitted += 1
        for a in agents:
            if not adversarial_at(a, rnd) and (a in excluded or a in flagged):
                honest_wrong_isolation[a] = honest_wrong_isolation.get(a, 0) + 1

        if admitted:
            if cfg["aggr"] is None:
                cc, ii = wavg([c["W"] for c in admitted],
                              [c["weight"] for c in admitted])
                fl._set_w(gm, cc, ii)
            else:
                coef, intc = cfg["aggr"](admitted)
                fl._set_w(gm, [coef], [intc])

        acc = float(gm.score(X_te, y_te))
        per_round.append({
            "round": rnd, "accuracy": round(acc, 4),
            "submitted": sorted(c["aid"] for c in submitted),
            "flagged": sorted(flagged), "excluded": sorted(excluded),
            "adv_admitted_cum": adv_admitted})

    readmitted = sorted(a for a in RETURNING
                        if a not in excluded and
                        any(r["round"] >= RETURN_ROUND and a in r["submitted"]
                            for r in per_round))
    return {
        "seed": seed, "condition": cond,
        "final_accuracy": per_round[-1]["accuracy"],
        "mean_accuracy_last10": round(
            st.mean(r["accuracy"] for r in per_round[-10:]), 4),
        "auc_accuracy": round(st.mean(r["accuracy"] for r in per_round), 4),
        "adv_updates_admitted": adv_admitted,
        "rounds_to_first_detection": {str(k): v - ATTACK_START
                                      for k, v in sorted(first_detect.items())},
        "rounds_to_isolation": {str(k): v - ATTACK_START
                                for k, v in sorted(first_isolate.items())},
        "honest_wrongly_isolated_rounds": {str(k): v for k, v in
                                           sorted(honest_wrong_isolation.items())},
        "returning_readmitted": readmitted,
        "final_excluded": sorted(excluded),
        "audit_reconstruction_rate": (round(recon_ok / recon_total, 4)
                                      if recon_total else None),
        "per_round": per_round}


def main(out_path, rounds=25, seed_start=72, seed_count=5):
    seeds = list(range(seed_start, seed_start + seed_count))
    results = {c: [] for c in CONDITIONS}
    for s in seeds:
        prep = fl.prepare(s)
        for cond in CONDITIONS:
            results[cond].append(run_condition(prep, s, cond, rounds))
        print("  seed %d done" % s, flush=True)

    print()
    hdr = "%-18s %9s %9s %12s %11s %11s"
    print(hdr % ("condition", "final acc", "AUC acc", "adv admitted",
                 "honest-wrong", "readmitted"))
    summary = {}
    for cond in CONDITIONS:
        rs = results[cond]
        hw = [sum(r["honest_wrongly_isolated_rounds"].values()) for r in rs]
        ra = [len(r["returning_readmitted"]) for r in rs]
        summary[cond] = {
            "final_accuracy": round(st.mean(r["final_accuracy"] for r in rs), 4),
            "final_accuracy_sd": round(st.pstdev([r["final_accuracy"] for r in rs]), 4),
            "auc_accuracy": round(st.mean(r["auc_accuracy"] for r in rs), 4),
            "adv_updates_admitted": round(st.mean(r["adv_updates_admitted"] for r in rs), 1),
            "honest_wrongly_isolated_rounds": round(st.mean(hw), 1),
            "returning_readmitted": round(st.mean(ra), 2),
            "audit_reconstruction_rate": (
                round(st.mean(r["audit_reconstruction_rate"] for r in rs), 4)
                if rs[0]["audit_reconstruction_rate"] is not None else None),
            "per_seed": rs}
        d = summary[cond]
        print(hdr % (cond, "%.4f" % d["final_accuracy"],
                     "%.4f" % d["auc_accuracy"],
                     "%.1f" % d["adv_updates_admitted"],
                     "%.1f" % d["honest_wrongly_isolated_rounds"],
                     "%.2f" % d["returning_readmitted"]))

    print()
    print("detection latency (rounds after attack start), seed %d:" % seeds[0])
    for cond in CONDITIONS:
        r0 = results[cond][0]
        print("  %-18s detect=%s isolate=%s" % (
            cond, r0["rounds_to_first_detection"], r0["rounds_to_isolation"]))

    Path(out_path).write_text(json.dumps(
        {"rounds": rounds, "seeds": seeds, "attack_start": ATTACK_START,
         "return_round": RETURN_ROUND, "returning": sorted(RETURNING),
         "summary": summary}, indent=1), encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(args[0],
                          int(args[1]) if len(args) > 1 else 25,
                          int(args[2]) if len(args) > 2 else 72,
                          int(args[3]) if len(args) > 3 else 5))
