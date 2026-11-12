# -*- coding: utf-8 -*-
"""E7 (P2-2): online adjudication as the federation grows.

The existing scalability experiment measured processing cost at up to 1,024
participants. It did not measure the two quantities that matter for governance
in live operation: how often an honest participant is wrongly isolated, and how
much adversarial update mass still reaches the aggregator.

The sample budget is scaled with the participant count so that each participant
keeps the same amount of local data (375 training samples). Holding the budget
fixed instead would drive per-participant data to about 23 samples at N=128,
which is at the usable floor the manuscript already reports, and would confound
"more participants" with "too little evidence per participant".

The adversarial fraction is held at 25%: one in four participants misbehaves,
cycling through label flipping, update scaling, replay and tampering, with one
further participant submitting under a revoked credential.

Conditions
  conv_norm      conventional checks + norm filter        (deployable)
  gov_no_sdf03   all 36 controls except S-DF-03           (deployable)
  gov_oracle     all 36 controls, S-DF-03 on true labels  (REFERENCE ONLY)

Usage
  python e7_scale.py run <n_agents> <cond> <seed> <out.jsonl>
  python e7_scale.py agg <dir> <out.json>
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[2]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
sys.path.insert(0, str(TB / "src"))

PER_AGENT_TRAIN = 375           # training samples per participant, held constant
ATTACK_START = 5
ROUNDS = 25
ADV_FRACTION = 0.25
BEHAVIOURS = ("label_flip", "scale_attack", "stale", "tamper")
FLIP_RATE = 0.70

# 20, 40 and 120 participants. The count must be at least 20 so that a 25%
# adversarial fraction still yields one participant of each of the four
# misbehaviour types plus one submitting under a revoked credential; at N=8 the
# same fraction would give only two adversarial roles, and the attack mix would
# then change with N alongside the participant count. The 8-participant
# configuration of the main experiments uses a different, far more adversarial
# profile (five of eight) and is deliberately not part of this comparison.
N_AGENTS = (20, 40, 120)
EVAL_SEEDS = list(range(500, 520))      # 20 seeds, disjoint from every other run
CONDS = ("conv_norm", "gov_no_sdf03", "gov_oracle")


def configure(n_agents):
    """Set the participant count and sample budget before trust is imported."""
    cap = int(round(n_agents * PER_AGENT_TRAIN / 0.75))
    os.environ["TRUST_NUM_AGENTS"] = str(n_agents)
    os.environ["TRUST_IMAGE_CAP"] = str(cap)
    return cap


def roles(n_agents):
    """Assign behaviours. Returns (profile, revoked_id, honest_ids)."""
    n_adv = int(round(n_agents * ADV_FRACTION))
    if n_adv < 5:
        raise SystemExit("n_agents=%d gives only %d adversaries; at least 5 are "
                         "needed to cover the four misbehaviour types plus one "
                         "revoked credential" % (n_agents, n_adv))
    revoked = n_agents - 1
    profile = {}
    for k in range(n_adv - 1):
        profile[k + 1] = BEHAVIOURS[k % len(BEHAVIOURS)]
    honest = [a for a in range(n_agents)
              if a not in profile and a != revoked]
    return profile, revoked, honest


def main_run(n_agents, cond, seed, out_path):
    cap = configure(n_agents)

    from trust import config                                          # noqa
    from trust import fl_simulation as fl                             # noqa
    from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa
    from trust.ledger import Ledger, sha256_hex, sign                 # noqa
    from trust.taxonomy import ALL_CONTROLS                           # noqa

    ALL_IDS = {cp.id for cp in ALL_CONTROLS}
    CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02",
                 "M-BC-01"}
    CONTROLS = {"conv_norm": CONV_NORM,
                "gov_no_sdf03": ALL_IDS - {"S-DF-03"},
                "gov_oracle": ALL_IDS}[cond]
    disabled = frozenset(ALL_IDS - CONTROLS)

    profile, revoked, honest_ids = roles(n_agents)

    prep = fl.prepare(seed, partition="iid")
    if "554" not in prep["dataset_name"] or prep["X_tr"].shape[1] != 784:
        raise SystemExit("wrong dataset: %s" % prep["dataset_name"])
    if config.NUM_AGENTS != n_agents:
        raise SystemExit("participant count not applied: %d != %d"
                         % (config.NUM_AGENTS, n_agents))

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

    def wavg(Ws, weights):
        w = np.asarray(weights, float)
        w /= w.sum()
        n_c, n_i = len(Ws[0][0]), len(Ws[0][1])
        return ([sum(w[k] * Ws[k][0][l] for k in range(len(Ws)))
                 for l in range(n_c)],
                [sum(w[k] * Ws[k][1][l] for k in range(len(Ws)))
                 for l in range(n_i)])

    def behaviour_at(aid, rnd):
        if rnd < ATTACK_START:
            return "honest"
        return profile.get(aid, "honest")

    def adversarial_at(aid, rnd):
        if rnd < ATTACK_START:
            return False
        return aid in profile or aid == revoked

    ledger = Ledger()
    ledger.reset()
    gm = fl._new_model(backbone, seed)
    gm.partial_fit(X_tr[:256], y_tr[:256], classes=classes)

    agents = {sh["agent_id"]: {
        "agent_id": sh["agent_id"], "principal": fl._principal(sh["agent_id"]),
        "behaviour": "honest", "is_adversary": False,
        "n_samples": int(len(sh["y"])), "rounds": []} for sh in shards}

    adv_admitted = 0
    adv_opportunities = 0
    honest_wrong_rounds = 0
    honest_ever_flagged = set()
    first_detect = {}
    per_round = []

    for rnd in range(ROUNDS):
        now = 1_700_000_000 + rnd * 3600
        gW = fl._get_w(gm)
        submitted = []
        for sh in shards:
            aid = sh["agent_id"]
            beh = behaviour_at(aid, rnd)
            Xl, yl = sh["X"].copy(), sh["y"].copy()
            if beh == "label_flip":
                yl = yl.copy()
                yl[rng.rand(len(yl)) < FLIP_RATE] = 0
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
                meta={"round": rnd, "behaviour": beh, "local_acc": 0.0,
                      "update_norm": round(update_norm, 6),
                      "label_kl": round(klv, 6), "logged": True,
                      "model_version": rnd + 1})
            agents[aid]["rounds"].append({
                "round": rnd, "update_norm": update_norm, "local_acc": 0.0,
                "label_kl": klv, "block_index": blk.index, "timestamp": ts,
                "ingest_time": now, "model_version": rnd + 1})
            agents[aid]["behaviour"] = beh
            agents[aid]["is_adversary"] = adversarial_at(aid, rnd)
            submitted.append({"W": (new_c, new_i), "weight": len(yl),
                              "aid": aid})

        signals = {
            "dataset": prep["dataset_name"], "num_agents": n_agents,
            "num_rounds": rnd + 1, "global_accuracy": 0.0,
            "revoked_agent": revoked if rnd >= ATTACK_START else None,
            "signing_scheme": config.SIGNING_SCHEME,
            "agents": {a: v for a, v in agents.items() if v["rounds"]},
            "seed": seed, "generated_at": now}
        ctx = build_ctx(signals)
        res = evaluate_controls(ctx)
        flagged = {a for a in ctx.agent_ids()
                   if agent_verdict(res, a, disabled=disabled)[0] != "Trusted"}

        for a in flagged:
            first_detect.setdefault(a, rnd)
        admitted = [c for c in submitted if c["aid"] not in flagged]
        for a in range(n_agents):
            if adversarial_at(a, rnd):
                adv_opportunities += 1
        for c in admitted:
            if adversarial_at(c["aid"], rnd):
                adv_admitted += 1
        for a in range(n_agents):
            if not adversarial_at(a, rnd) and a in flagged:
                honest_wrong_rounds += 1
                honest_ever_flagged.add(a)

        if admitted:
            cc, ii = wavg([c["W"] for c in admitted],
                          [c["weight"] for c in admitted])
            fl._set_w(gm, cc, ii)
        per_round.append({"round": rnd,
                          "accuracy": round(float(gm.score(X_te, y_te)), 4),
                          "n_flagged": len(flagged),
                          "adv_admitted_cum": adv_admitted})

    n_honest = len(honest_ids)
    honest_slots = n_honest * ROUNDS
    adv_detected = sorted(a for a in first_detect
                          if a in profile or a == revoked)
    rec = {
        "n_agents": n_agents, "condition": cond, "seed": seed,
        "sample_cap": cap, "per_agent_train": PER_AGENT_TRAIN,
        "samples_per_agent_actual": int(st.mean(
            [a["n_samples"] for a in agents.values()])),
        "n_adversaries": len(profile) + 1, "n_honest": n_honest,
        "adv_fraction": round((len(profile) + 1) / n_agents, 4),
        "rounds": ROUNDS,
        "final_accuracy": per_round[-1]["accuracy"],
        "auc_accuracy": round(st.mean(r["accuracy"] for r in per_round), 4),
        "adv_admitted": adv_admitted,
        "adv_opportunities": adv_opportunities,
        "adv_admitted_share": round(adv_admitted / max(adv_opportunities, 1), 4),
        "adversary_recall": round(len(adv_detected) / (len(profile) + 1), 4),
        "honest_wrong_rounds": honest_wrong_rounds,
        "honest_wrong_rate": round(honest_wrong_rounds / max(honest_slots, 1), 4),
        "honest_ever_flagged": len(honest_ever_flagged),
        "honest_fpr": round(len(honest_ever_flagged) / max(n_honest, 1), 4),
        "accuracy_per_round": [r["accuracy"] for r in per_round],
    }
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


def boot_ci(vals, n=2000, seed=0):
    if not vals:
        return None
    rs = np.random.RandomState(seed)
    a = np.asarray(vals, float)
    bs = a[rs.randint(0, len(a), size=(n, len(a)))].mean(axis=1)
    return [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def mode_agg(d, out_path):
    rows = []
    for p in sorted(Path(d).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no records in %s" % d)
    summary = {}
    for n in N_AGENTS:
        for cond in CONDS:
            rs = [r for r in rows if r["n_agents"] == n
                  and r["condition"] == cond]
            if not rs:
                continue
            summary["N%d|%s" % (n, cond)] = {
                "n_agents": n, "condition": cond, "n_seeds": len(rs),
                "sample_cap": rs[0]["sample_cap"],
                "samples_per_agent": rs[0]["samples_per_agent_actual"],
                "n_adversaries": rs[0]["n_adversaries"],
                "adv_fraction": rs[0]["adv_fraction"],
                "adv_admitted_share_mean": round(st.mean(
                    r["adv_admitted_share"] for r in rs), 4),
                "adv_admitted_share_ci": boot_ci(
                    [r["adv_admitted_share"] for r in rs]),
                "adversary_recall_mean": round(st.mean(
                    r["adversary_recall"] for r in rs), 4),
                "honest_fpr_mean": round(st.mean(r["honest_fpr"] for r in rs), 4),
                "honest_fpr_ci": boot_ci([r["honest_fpr"] for r in rs]),
                "honest_wrong_rate_mean": round(st.mean(
                    r["honest_wrong_rate"] for r in rs), 4),
                "final_accuracy_mean": round(st.mean(
                    r["final_accuracy"] for r in rs), 4),
                "final_accuracy_ci": boot_ci([r["final_accuracy"] for r in rs]),
            }
    Path(out_path).write_text(json.dumps({
        "experiment": "E7 / P2-2 online adjudication vs participant count",
        "evaluation_seeds": EVAL_SEEDS, "n_agents": list(N_AGENTS),
        "conditions": list(CONDS), "rounds": ROUNDS,
        "attack_start": ATTACK_START, "adv_fraction_target": ADV_FRACTION,
        "per_agent_train_samples": PER_AGENT_TRAIN, "flip_rate": FLIP_RATE,
        "n_records": len(rows), "summary": summary,
    }, indent=1), encoding="utf-8")
    print("wrote", out_path, "from", len(rows), "records")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "run":
        raise SystemExit(main_run(int(a[1]), a[2], int(a[3]), a[4]))
    if a[0] == "agg":
        raise SystemExit(mode_agg(a[1], a[2]))
    raise SystemExit("unknown mode %r" % a[0])
