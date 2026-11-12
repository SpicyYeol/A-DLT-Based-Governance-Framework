# -*- coding: utf-8 -*-
"""E5 (P0-3): enforcement policies compared on the same evidence.

The online-adjudication comparison (Table 11 of the paper) showed that making
exclusions permanent triples the participant-rounds
in which an honest participant is excluded (9.4 to 27.6) and never readmits a
participant that returns to honest behaviour. Permanent exclusion is therefore
not a defensible default, and the question is what to put in its place.

Five policies are compared, each applied to both control sets so the comparison
is fair:

  A per_round        adjudicate independently every round, no memory
  B persist_any      any failed control excludes the participant permanently
  C persist_determ   only a deterministic (cryptographic, identity, freshness,
                     integrity) Reject excludes permanently; a statistical
                     finding excludes for the current round only
  D review_downweight  deterministic Reject excludes permanently; a statistical
                     finding is Review severity and down-weights the
                     contribution by lambda instead of dropping it, and the
                     participant is re-evaluated every round
  E two_strikes      deterministic Reject excludes permanently; a statistical
                     finding quarantines only after two consecutive rounds of
                     statistical violation

The statistical family is {M-BC-01, M-BC-02, M-SC-02, S-DF-03}, as in the
control-family experiment. For the conventional control set the only
statistical member is M-BC-01, so the same policy definitions apply unchanged.

Usage
  python e5_policy.py run <cell> <ctrlset> <policy> <seed> <out.jsonl>
  python e5_policy.py agg <eval_dir> <out.json>
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
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402
from trust.ledger import Ledger, sha256_hex, sign                     # noqa: E402
from trust.taxonomy import ALL_CONTROLS                               # noqa: E402

ALL_IDS = {cp.id for cp in ALL_CONTROLS}
STATISTICAL = {"M-BC-01", "M-BC-02", "M-SC-02", "S-DF-03"}
CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02", "M-BC-01"}

CTRLSETS = {"gov": ALL_IDS, "conv": CONV_NORM}
POLICIES = ("A_per_round", "B_persist_any", "C_persist_determ",
            "D_review_downweight", "E_two_strikes")
REVIEW_LAMBDA = 0.25          # down-weight multiplier, as in the quarantine
                              # comparison of Section 6.1

ATTACK_START = 5
RETURN_ROUND = 15
RETURNING = {5, 6}
REVOKED = 7
BASE_PROFILE = {1: "label_flip", 3: "scale_attack", 5: "stale", 6: "tamper"}
ALWAYS_HONEST = (0, 2, 4)

EVAL_SEEDS = list(range(300, 350))     # 50 seeds, disjoint from every other run

CELLS = {
    "r25_iid": dict(rounds=25, partition="iid", alpha=1.0, flip=0.70),
    "r25_dir": dict(rounds=25, partition="dirichlet", alpha=0.5, flip=0.70),
    "r50_iid": dict(rounds=50, partition="iid", alpha=1.0, flip=0.70),
    "r50_dir": dict(rounds=50, partition="dirichlet", alpha=0.5, flip=0.70),
}


def behaviour_at(aid, rnd):
    if rnd < ATTACK_START:
        return "honest"
    if aid in RETURNING and rnd >= RETURN_ROUND:
        return "honest"
    return BASE_PROFILE.get(aid, "honest")


def adversarial_at(aid, rnd):
    if rnd < ATTACK_START:
        return False
    if aid == REVOKED:
        return True
    if aid in RETURNING and rnd >= RETURN_ROUND:
        return False
    return aid in BASE_PROFILE


def adversarial_opportunities(rounds):
    return sum(1 for r in range(rounds) for a in range(config.NUM_AGENTS)
               if adversarial_at(a, r))


def run(cell_key, ctrlset_key, policy, seed):
    cell = CELLS[cell_key]
    rounds, flip = cell["rounds"], cell["flip"]
    controls = CTRLSETS[ctrlset_key]
    stat_here = STATISTICAL & controls
    disabled = frozenset(ALL_IDS - controls)

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

    excluded = set()                   # permanent
    stat_streak = {a: 0 for a in agents}
    adv_admitted = 0.0                 # weighted: a down-weighted update counts as lambda
    adv_admitted_raw = 0               # unweighted count of adversarial updates aggregated
    first_detect, first_enforce = {}, {}
    honest_wrong_rounds = {}           # aid -> number of rounds wrongly excluded
    honest_downweighted = {}           # aid -> rounds wrongly down-weighted
    readmit_round = {}                 # returning aid -> first round readmitted
    per_round = []

    for rnd in range(rounds):
        now = 1_700_000_000 + rnd * 3600
        gW = fl._get_w(gm)
        submitted = []
        for sh in shards:
            aid = sh["agent_id"]
            if aid in excluded:
                continue
            beh = behaviour_at(aid, rnd)
            Xl, yl = sh["X"].copy(), sh["y"].copy()
            if beh in ("label_flip", "collude_flip"):
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
            agents[aid]["is_adversary"] = adversarial_at(aid, rnd)
            submitted.append({"W": (new_c, new_i), "weight": float(len(yl)),
                              "aid": aid})

        # ---- adjudicate on the evidence recorded so far -------------------
        signals = {
            "dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
            "num_rounds": rnd + 1, "global_accuracy": 0.0,
            "revoked_agent": REVOKED if rnd >= ATTACK_START else None,
            "signing_scheme": config.SIGNING_SCHEME,
            "agents": {a: v for a, v in agents.items() if v["rounds"]},
            "seed": seed, "generated_at": now}
        ctx = build_ctx(signals)
        res = evaluate_controls(ctx)

        failed_by = {}
        for a in ctx.agent_ids():
            if a in excluded:
                continue
            verdict, fired = agent_verdict(res, a, disabled=disabled)
            if verdict != "Trusted":
                failed_by[a] = fired

        # ---- translate findings into an action under the active policy ----
        drop, downweight, newly_permanent = set(), set(), set()
        for a, fired in failed_by.items():
            determ_reject = any(sev == "reject" and cid not in stat_here
                                for cid, sev in fired.items())
            stat_fired = bool(set(fired) & stat_here)
            stat_streak[a] = stat_streak[a] + 1 if stat_fired else 0

            if policy == "A_per_round":
                drop.add(a)
            elif policy == "B_persist_any":
                drop.add(a)
                newly_permanent.add(a)
            elif policy == "C_persist_determ":
                drop.add(a)
                if determ_reject:
                    newly_permanent.add(a)
            elif policy == "D_review_downweight":
                if determ_reject:
                    drop.add(a)
                    newly_permanent.add(a)
                else:
                    downweight.add(a)
            elif policy == "E_two_strikes":
                if determ_reject:
                    drop.add(a)
                    newly_permanent.add(a)
                elif stat_streak[a] >= 2:
                    drop.add(a)
                    newly_permanent.add(a)
                else:
                    downweight.add(a)
            else:
                raise SystemExit("unknown policy %r" % policy)

        for a in failed_by:
            first_detect.setdefault(a, rnd)
        for a in drop | downweight:
            first_enforce.setdefault(a, rnd)

        kept = []
        for c in submitted:
            a = c["aid"]
            if a in drop:
                continue
            w = c["weight"] * (REVIEW_LAMBDA if a in downweight else 1.0)
            kept.append({"W": c["W"], "weight": w, "aid": a,
                         "share": REVIEW_LAMBDA if a in downweight else 1.0})

        for c in kept:
            if adversarial_at(c["aid"], rnd):
                adv_admitted += c["share"]
                adv_admitted_raw += 1

        for a in agents:
            if adversarial_at(a, rnd):
                continue
            if a in excluded or a in drop:
                honest_wrong_rounds[a] = honest_wrong_rounds.get(a, 0) + 1
            elif a in downweight:
                honest_downweighted[a] = honest_downweighted.get(a, 0) + 1

        excluded |= newly_permanent

        for a in RETURNING:
            if (rnd >= RETURN_ROUND and a not in excluded
                    and a in [c["aid"] for c in submitted]
                    and a not in drop and a not in readmit_round):
                readmit_round[a] = rnd

        if kept:
            cc, ii = wavg([c["W"] for c in kept], [c["weight"] for c in kept])
            fl._set_w(gm, cc, ii)

        acc = float(gm.score(X_te, y_te))
        per_round.append({
            "round": rnd, "accuracy": round(acc, 4),
            "submitted": sorted(c["aid"] for c in submitted),
            "dropped": sorted(drop), "downweighted": sorted(downweight),
            "excluded": sorted(excluded),
            "adv_admitted_cum": round(adv_admitted, 3)})

    honest_round_slots = len(ALWAYS_HONEST) * rounds
    honest_kept = honest_round_slots - sum(
        v for a, v in honest_wrong_rounds.items() if a in ALWAYS_HONEST)

    return {
        "cell": cell_key, "ctrlset": ctrlset_key, "policy": policy,
        "seed": seed, "rounds": rounds, "flip": flip,
        "partition": cell["partition"], "alpha": cell["alpha"],
        "review_lambda": REVIEW_LAMBDA,
        "final_accuracy": per_round[-1]["accuracy"],
        "auc_accuracy": round(st.mean(r["accuracy"] for r in per_round), 4),
        "adv_opportunities": adversarial_opportunities(rounds),
        "adv_admitted_weighted": round(adv_admitted, 3),
        "adv_admitted_raw": adv_admitted_raw,
        "honest_wrong_rounds": sum(honest_wrong_rounds.values()),
        "honest_wrong_rounds_by_agent": {str(k): v for k, v in
                                         sorted(honest_wrong_rounds.items())},
        "honest_downweighted_rounds": sum(honest_downweighted.values()),
        "honest_retention": round(honest_kept / honest_round_slots, 4),
        "returning_readmitted": sorted(readmit_round),
        "rounds_to_readmission": {str(a): readmit_round[a] - RETURN_ROUND
                                  for a in sorted(readmit_round)},
        "rounds_to_first_detection": {str(k): v - ATTACK_START
                                      for k, v in sorted(first_detect.items())},
        "rounds_to_enforcement": {str(k): v - ATTACK_START
                                  for k, v in sorted(first_enforce.items())},
        "final_excluded": sorted(excluded),
        "accuracy_per_round": [r["accuracy"] for r in per_round],
    }


def boot_ci(vals, n=2000, seed=0):
    if not vals:
        return None
    rs = np.random.RandomState(seed)
    a = np.asarray(vals, float)
    bs = a[rs.randint(0, len(a), size=(n, len(a)))].mean(axis=1)
    return [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def mode_run(cell, ctrlset, policy, seed, out_path):
    r = run(cell, ctrlset, policy, int(seed))
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n")
    return 0


def mode_agg(eval_dir, out_path):
    rows = []
    for p in sorted(Path(eval_dir).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no records in %s" % eval_dir)

    summary = {}
    for cell in CELLS:
        for cs in CTRLSETS:
            for pol in POLICIES:
                rs = [r for r in rows if r["cell"] == cell
                      and r["ctrlset"] == cs and r["policy"] == pol]
                if not rs:
                    continue
                adv = [r["adv_admitted_weighted"] for r in rs]
                acc = [r["final_accuracy"] for r in rs]
                hw = [r["honest_wrong_rounds"] for r in rs]
                hd = [r["honest_downweighted_rounds"] for r in rs]
                hr = [r["honest_retention"] for r in rs]
                ra = [len(r["returning_readmitted"]) for r in rs]
                lat = [v for r in rs for v in r["rounds_to_readmission"].values()]
                summary["%s|%s|%s" % (cell, cs, pol)] = {
                    "cell": cell, "ctrlset": cs, "policy": pol,
                    "n_seeds": len(rs),
                    "adv_opportunities": rs[0]["adv_opportunities"],
                    "adv_admitted_weighted_mean": round(st.mean(adv), 2),
                    "adv_admitted_weighted_ci": boot_ci(adv),
                    "adv_admitted_raw_mean": round(
                        st.mean(r["adv_admitted_raw"] for r in rs), 2),
                    "final_accuracy_mean": round(st.mean(acc), 4),
                    "final_accuracy_sd": round(st.pstdev(acc), 4),
                    "final_accuracy_ci": boot_ci(acc),
                    "honest_wrong_rounds_mean": round(st.mean(hw), 2),
                    "honest_wrong_rounds_ci": boot_ci(hw),
                    "honest_downweighted_rounds_mean": round(st.mean(hd), 2),
                    "honest_retention_mean": round(st.mean(hr), 4),
                    "returning_readmitted_mean": round(st.mean(ra), 2),
                    "rounds_to_readmission_mean": (round(st.mean(lat), 2)
                                                   if lat else None),
                }
    Path(out_path).write_text(json.dumps({
        "experiment": "E5 / P0-3 enforcement policy comparison",
        "evaluation_seeds": EVAL_SEEDS, "cells": CELLS,
        "policies": list(POLICIES), "control_sets": list(CTRLSETS),
        "statistical_family": sorted(STATISTICAL),
        "review_lambda": REVIEW_LAMBDA,
        "attack_start": ATTACK_START, "return_round": RETURN_ROUND,
        "returning": sorted(RETURNING), "revoked": REVOKED,
        "n_records": len(rows), "summary": summary,
    }, indent=1), encoding="utf-8")
    print("wrote", out_path, "from", len(rows), "records")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        raise SystemExit(mode_run(a[1], a[2], a[3], a[4], a[5]))
    if a[0] == "agg":
        raise SystemExit(mode_agg(a[1], a[2]))
    raise SystemExit("unknown mode %r" % a[0])
