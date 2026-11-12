# -*- coding: utf-8 -*-
"""Does governance still beat the conventional baseline when the server may
use only information it actually holds?

The online-adjudication comparison (Table 11 of the paper) credited the full
control set with cutting adversarial admissions from 20.6 to 13.2, but the
control responsible for the difference, S-DF-03, reads a
label distribution a deployed server does not have. This experiment replaces
that privileged input with each channel a deployment could really use and
re-runs the online comparison.

Conditions
  conv_norm          conventional security checks + the norm filter, per round
  conv_norm_persist  the same, with persistent exclusion
  gov_oracle         all 36 controls, S-DF-03 on true labels   (REFERENCE ONLY,
                     not deployable -- the published rule)
  gov_no_sdf03       all 36 controls except S-DF-03            (deployable)
  gov_grad           all 36, S-DF-03 fed by the update-derived estimate
  gov_anchor         all 36, S-DF-03 fed by a server-anchored direction score
  fltrust            FLTrust aggregation, no verdicts          (reference)

gov_anchor and fltrust both use the same trusted root set (10 samples per
class), so the privileged-data advantage is granted equally to the comparison
method, as a fair baseline requires.

Channel substitution. S-DF-03 flags an agent when its mean label KL exceeds 0.60.
A channel is substituted by writing, in place of the true label KL for that
round, ``0.20 * score / tau`` where ``score`` is the channel's own per-round
suspicion score and ``tau`` its frozen threshold. The mean of that quantity
exceeds 0.20 exactly when the mean channel score exceeds tau, so the published
rule is applied unchanged to the substituted signal rather than replaced by a
different rule.

Thresholds are fitted on calibration seeds and frozen to a file before any
evaluation seed is touched; the evaluation mode refuses to invent one.

Usage
  python e4_deployable.py calib  <cell> <seed> <out.jsonl>
  python e4_deployable.py freeze <calib_dir> <out_thresholds.json>
  python e4_deployable.py eval   <cell> <cond> <seed> <thresholds.json> <out.jsonl>
  python e4_deployable.py agg    <eval_dir> <thresholds.json> <out.json>
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
from trust import fltrust                                             # noqa: E402
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402
from trust.ledger import Ledger, sha256_hex, sign                     # noqa: E402
from trust.taxonomy import ALL_CONTROLS                               # noqa: E402

EPS = 1e-12
# S-DF-03 scores an agent as 1.0 for mean label KL <= 0.20 and degrades
# linearly to 0 at KL = 1.0; agent_verdict flags a control whose per-agent
# subscore falls below 0.5, which happens at mean KL > 0.60. The enforced
# cutoff is therefore 0.60, not the 0.20 at which the score starts to degrade.
SDF03_DEGRADE_FROM = 0.20
SDF03_FAIL_KL = 0.60
ROOT_PER_CLASS = 10                   # trusted root set size per class

ALL_IDS = {cp.id for cp in ALL_CONTROLS}
CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02", "M-BC-01"}

ATTACK_START = 5
RETURN_ROUND = 15
RETURNING = {5, 6}                    # replay and tampering revert to honest
REVOKED = 7
BASE_PROFILE = {1: "label_flip", 3: "scale_attack", 5: "stale", 6: "tamper"}
FLIPPER = 1
ALWAYS_HONEST = (0, 2, 4)

# Seed ranges. 42-71, 72-76, 77-101, 102-106 and 110-129 are all consumed by
# earlier experiments; these two ranges have never been evaluated.
CALIB_SEEDS = list(range(150, 160))   # 10 seeds, thresholds fitted here
EVAL_SEEDS = list(range(200, 250))    # 50 seeds, never inspected before freeze

CELLS = {
    "r25_iid_f30": dict(rounds=25, partition="iid", alpha=1.0, flip=0.30),
    "r25_iid_f70": dict(rounds=25, partition="iid", alpha=1.0, flip=0.70),
    "r25_dir_f30": dict(rounds=25, partition="dirichlet", alpha=0.5, flip=0.30),
    "r25_dir_f70": dict(rounds=25, partition="dirichlet", alpha=0.5, flip=0.70),
    "r50_iid_f30": dict(rounds=50, partition="iid", alpha=1.0, flip=0.30),
    "r50_iid_f70": dict(rounds=50, partition="iid", alpha=1.0, flip=0.70),
    "r50_dir_f30": dict(rounds=50, partition="dirichlet", alpha=0.5, flip=0.30),
    "r50_dir_f70": dict(rounds=50, partition="dirichlet", alpha=0.5, flip=0.70),
}

CONDITIONS = {
    "conv_norm":         dict(controls=CONV_NORM, persist=False, channel=None,
                              aggr="fedavg", deployable=True),
    "conv_norm_persist": dict(controls=CONV_NORM, persist=True, channel=None,
                              aggr="fedavg", deployable=True),
    "gov_oracle":        dict(controls=ALL_IDS, persist=False, channel="oracle",
                              aggr="fedavg", deployable=False),
    "gov_no_sdf03":      dict(controls=ALL_IDS - {"S-DF-03"}, persist=False,
                              channel=None, aggr="fedavg", deployable=True),
    "gov_grad":          dict(controls=ALL_IDS, persist=False, channel="gradient",
                              aggr="fedavg", deployable=True),
    "gov_anchor":        dict(controls=ALL_IDS, persist=False,
                              channel="server_anchored", aggr="fedavg",
                              deployable=True),
    "fltrust":           dict(controls=None, persist=False, channel=None,
                              aggr="fltrust", deployable=True),
}
CHANNEL_CONDS = ("gradient", "server_anchored")


# --------------------------------------------------------------------------- #
# channel scores
# --------------------------------------------------------------------------- #
def kl(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return float(np.sum(np.where(p > 0, p * np.log((p + EPS) / (q + EPS)), 0.0)))


def estimate_dist(delta_b, prior):
    """Recover a label distribution from the output-layer bias update.

    For a softmax output layer the per-class bias gradient is a function of the
    difference between observed and predicted class frequencies, so the update
    direction carries P_a up to a positive scale. kappa is the largest step that
    keeps every recovered probability non-negative.
    """
    d = np.asarray(delta_b, float) - np.mean(delta_b)
    if np.allclose(d, 0):
        return np.asarray(prior, float)
    neg = d < 0
    kappa = float(np.min(-np.asarray(prior)[neg] / d[neg])) if neg.any() else 1.0
    if not np.isfinite(kappa) or kappa <= 0:
        kappa = 1.0
    est = np.clip(np.asarray(prior, float) + kappa * d, 0.0, None)
    tot = est.sum()
    return est / tot if tot > 0 else np.asarray(prior, float)


def flat(W):
    """Flatten a (coefs, intercepts) weight pair into one vector."""
    return np.concatenate([np.asarray(a, float).ravel() for a in W[0]]
                          + [np.asarray(b, float).ravel() for b in W[1]])


# --------------------------------------------------------------------------- #
# behaviour schedule
# --------------------------------------------------------------------------- #
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
    """Ground-truth number of adversarial submissions available over a run."""
    return sum(1 for r in range(rounds) for a in range(config.NUM_AGENTS)
               if adversarial_at(a, r))


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def run(cell_key, cond_key, seed, taus, collect_scores=False):
    """Execute one online run. ``taus`` may be None only when no channel is used.

    Returns a result dict. With ``collect_scores`` the per-agent mean channel
    scores are returned as well, which is what calibration needs.
    """
    cell = CELLS[cell_key]
    cfg = CONDITIONS[cond_key] if cond_key else dict(
        controls=None, persist=False, channel=None, aggr="fedavg",
        deployable=True)
    rounds, flip = cell["rounds"], cell["flip"]
    channel = cfg["channel"]
    if channel in CHANNEL_CONDS:
        if not taus or channel not in taus:
            raise SystemExit("evaluation needs a frozen threshold for %s" % channel)
        tau = float(taus[channel][cell_key])
    else:
        tau = None

    prep = fl.prepare(seed, partition=cell["partition"], alpha=cell["alpha"])
    # The image loader falls back to sklearn's bundled 8x8 digits on any fetch
    # failure. That would silently produce numbers from a different dataset, so
    # fail the task instead.
    if "554" not in prep["dataset_name"] or prep["X_tr"].shape[1] != 784:
        raise SystemExit("wrong dataset: %s shape=%s"
                         % (prep["dataset_name"], prep["X_tr"].shape))
    rng = np.random.RandomState(seed)
    backbone = prep.get("backbone", "linear")
    X_tr, y_tr = prep["X_tr"], prep["y_tr"]
    X_te, y_te = prep["X_te"], prep["y_te"]
    shards, base_dist = prep["shards"], prep["base_dist"]
    classes = np.unique(y_tr)
    ncls = prep["n_classes"]

    # trusted root set: the privileged data that gov_anchor and fltrust share
    root_idx = []
    for c in range(ncls):
        root_idx.extend(np.where(y_tr == c)[0][:ROOT_PER_CLASS].tolist())
    root_idx = np.asarray(root_idx)
    X_root, y_root = X_tr[root_idx], y_tr[root_idx]

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

    excluded = set()
    adv_admitted = 0
    first_detect, first_isolate = {}, {}
    honest_wrong = {}
    detected_rounds = {}          # aid -> rounds in which it was flagged
    adv_rounds = {}               # aid -> rounds in which it was adversarial
    score_hist = {a: {"gradient": [], "server_anchored": [], "oracle": [],
                      "norm": []} for a in agents}
    per_round = []
    recon_ok = recon_total = 0

    for rnd in range(rounds):
        now = 1_700_000_000 + rnd * 3600
        gW = fl._get_w(gm)

        # the server's own root update, from the current global model
        sW = fl._get_w(local_fit(gW, X_root, y_root))
        server_delta = flat(sW) - flat(gW)

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
            elif beh == "stealth_scale":
                f = config.STEALTH_SCALE_FACTOR
                d_c = [x * f for x in d_c]
                d_i = [x * f for x in d_i]

            new_c = [g + d for g, d in zip(gW[0], d_c)]
            new_i = [g + d for g, d in zip(gW[1], d_i)]
            delta_flat = flat((d_c, d_i))

            update_norm = float(sum(np.linalg.norm(a) for a in d_c)
                                + sum(np.linalg.norm(a) for a in d_i))
            local_acc = float(m.score(X_te, y_te))

            # ---- the three label-distribution channels ---------------------
            kl_oracle = fl._label_kl(yl, ncls, base_dist)
            kl_grad = kl(estimate_dist(delta_flat[-ncls:], base_dist), base_dist)
            n_a = float(np.linalg.norm(delta_flat)) or EPS
            n_s = float(np.linalg.norm(server_delta)) or EPS
            cos = float(np.dot(delta_flat, server_delta) / (n_a * n_s))
            anchor_score = 1.0 - max(0.0, cos)       # higher = more suspicious

            score_hist[aid]["oracle"].append(kl_oracle)
            score_hist[aid]["gradient"].append(kl_grad)
            score_hist[aid]["server_anchored"].append(anchor_score)
            score_hist[aid]["norm"].append(update_norm)

            # Substituting a channel: write a pseudo-KL that is a positive
            # multiple of the channel score, scaled so that the control's own
            # enforced rule (mean pseudo-KL > SDF03_FAIL_KL) fires exactly when
            # the mean channel score exceeds that channel's frozen threshold.
            if channel == "oracle" or channel is None:
                klv = kl_oracle
            elif channel == "gradient":
                klv = SDF03_FAIL_KL * kl_grad / max(tau, EPS)
            else:
                klv = SDF03_FAIL_KL * anchor_score / max(tau, EPS)

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
                              "aid": aid, "delta": delta_flat,
                              "coef": new_c[0], "int": new_i[0]})

        # ---- adjudication on the evidence recorded so far ------------------
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

            blocks = Ledger().read()
            rebuilt = {}
            for b in blocks:
                rebuilt.setdefault(int(b["agent_id"]), []).append(b.get("meta", {}))
            for a in ctx.agent_ids():
                recon_total += 1
                if a in rebuilt and len(rebuilt[a]) == len(agents[a]["rounds"]):
                    recon_ok += 1

        for a in flagged:
            first_detect.setdefault(a, rnd)
            detected_rounds.setdefault(a, []).append(rnd)
        for a in agents:
            if adversarial_at(a, rnd):
                adv_rounds.setdefault(a, []).append(rnd)

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
                honest_wrong[a] = honest_wrong.get(a, 0) + 1

        if admitted:
            if cfg["aggr"] == "fltrust":
                # FLTrust: weight by ReLU-cosine to the root update and rescale
                # every accepted update to the root update's norm.
                sc = fltrust.fltrust_scores(
                    {c["aid"]: c["delta"] for c in admitted}, server_delta)
                tot = sum(sc[c["aid"]]["relu_cos"] for c in admitted)
                if tot <= 0:
                    agg_delta = np.zeros_like(server_delta)
                else:
                    agg_delta = np.zeros_like(server_delta)
                    for c in admitted:
                        w = sc[c["aid"]]["relu_cos"] / tot
                        nd = float(np.linalg.norm(c["delta"])) or EPS
                        agg_delta += w * c["delta"] * (n_s / nd)
                newflat = flat(gW) + agg_delta
                n_coef = sum(np.asarray(a).size for a in gW[0])
                cc = [newflat[:n_coef].reshape(np.asarray(gW[0][0]).shape)]
                ii = [newflat[n_coef:]]
                fl._set_w(gm, cc, ii)
            else:
                cc, ii = wavg([c["W"] for c in admitted],
                              [c["weight"] for c in admitted])
                fl._set_w(gm, cc, ii)

        acc = float(gm.score(X_te, y_te))
        per_round.append({
            "round": rnd, "accuracy": round(acc, 4),
            "submitted": sorted(c["aid"] for c in submitted),
            "flagged": sorted(flagged), "excluded": sorted(excluded),
            "adv_admitted_cum": adv_admitted})

    readmitted = sorted(a for a in RETURNING
                        if a not in excluded
                        and any(r["round"] >= RETURN_ROUND and a in r["submitted"]
                                for r in per_round))

    # per-attacker detection rate: adversarial rounds in which it was flagged
    det_rate = {}
    for a, ars in adv_rounds.items():
        hits = set(detected_rounds.get(a, []))
        det_rate[str(a)] = round(
            sum(1 for r in ars if r in hits) / max(len(ars), 1), 4)

    mean_scores = {ch: {str(a): round(float(st.mean(v[ch])), 6)
                        for a, v in score_hist.items() if v[ch]}
                   for ch in ("oracle", "gradient", "server_anchored", "norm")}

    out = {
        "cell": cell_key, "condition": cond_key, "seed": seed,
        "rounds": rounds, "flip": flip, "partition": cell["partition"],
        "alpha": cell["alpha"], "tau": tau,
        "deployable": cfg.get("deployable"),
        "final_accuracy": per_round[-1]["accuracy"],
        "auc_accuracy": round(st.mean(r["accuracy"] for r in per_round), 4),
        "accuracy_per_round": [r["accuracy"] for r in per_round],
        "adv_updates_admitted": adv_admitted,
        "adv_opportunities": adversarial_opportunities(rounds),
        "per_attacker_detection_rate": det_rate,
        "honest_wrongly_isolated_rounds": sum(honest_wrong.values()),
        "honest_wrongly_isolated_by_agent": {str(k): v for k, v in
                                             sorted(honest_wrong.items())},
        "honest_agents_ever_flagged": sorted(
            a for a in ALWAYS_HONEST if a in first_detect),
        "rounds_to_first_detection": {str(k): v - ATTACK_START
                                      for k, v in sorted(first_detect.items())},
        "rounds_to_isolation": {str(k): v - ATTACK_START
                                for k, v in sorted(first_isolate.items())},
        "returning_readmitted": readmitted,
        "final_excluded": sorted(excluded),
        "audit_reconstruction_rate": (round(recon_ok / recon_total, 4)
                                      if recon_total else None),
    }
    if collect_scores:
        out["mean_scores"] = mean_scores
    return out


# --------------------------------------------------------------------------- #
# threshold fitting
# --------------------------------------------------------------------------- #
def pick_threshold(vals_pos, vals_neg):
    """Midpoint when separable, otherwise the cut maximising Youden's J."""
    if vals_pos and vals_neg and min(vals_pos) > max(vals_neg):
        return (min(vals_pos) + max(vals_neg)) / 2.0
    cand = sorted(set(list(vals_pos) + list(vals_neg)))
    best, best_j = (cand[0] if cand else 0.0), -2.0
    for c in cand:
        tpr = sum(1 for v in vals_pos if v > c) / max(len(vals_pos), 1)
        fpr = sum(1 for v in vals_neg if v > c) / max(len(vals_neg), 1)
        if tpr - fpr > best_j:
            best_j, best = tpr - fpr, c
    return best


def auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def mode_calib(cell_key, seed, out_path):
    """One calibration run: no adjudication, channel scores only."""
    r = run(cell_key, None, seed, taus=None, collect_scores=True)
    rec = {"cell": cell_key, "seed": seed, "mean_scores": r["mean_scores"]}
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


def mode_freeze(calib_dir, out_path):
    """Fit one threshold per (channel, cell) on the calibration seeds only."""
    recs = []
    for p in sorted(Path(calib_dir).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit("no calibration records in %s" % calib_dir)

    taus = {ch: {} for ch in CHANNEL_CONDS}
    detail = {}
    for cell_key in CELLS:
        rows = [r for r in recs if r["cell"] == cell_key]
        if not rows:
            raise SystemExit("no calibration rows for cell %s" % cell_key)
        for ch in CHANNEL_CONDS:
            pos = [r["mean_scores"][ch][str(FLIPPER)] for r in rows
                   if str(FLIPPER) in r["mean_scores"][ch]]
            neg = [r["mean_scores"][ch][str(a)] for r in rows
                   for a in ALWAYS_HONEST if str(a) in r["mean_scores"][ch]]
            t = pick_threshold(pos, neg)
            taus[ch][cell_key] = round(float(t), 6)
            detail.setdefault(cell_key, {})[ch] = {
                "threshold": round(float(t), 6),
                "flipper_scores": [round(v, 6) for v in sorted(pos)],
                "honest_min": round(min(neg), 6) if neg else None,
                "honest_max": round(max(neg), 6) if neg else None,
                "calibration_auroc": auroc(pos, neg),
                "separable": bool(pos and neg and min(pos) > max(neg)),
            }
    payload = {"calibration_seeds": sorted({r["seed"] for r in recs}),
               "n_calibration_records": len(recs),
               "positives": [FLIPPER], "negatives": list(ALWAYS_HONEST),
               "sdf03_degrade_from": SDF03_DEGRADE_FROM,
               "sdf03_enforced_fail_kl": SDF03_FAIL_KL,
               "thresholds": taus, "detail": detail}
    Path(out_path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("frozen thresholds written to", out_path)
    for cell_key in CELLS:
        print("  %-12s %s" % (cell_key,
                              {ch: taus[ch][cell_key] for ch in CHANNEL_CONDS}))
    return 0


def mode_eval(cell_key, cond_key, seed, thr_path, out_path):
    thr = json.loads(Path(thr_path).read_text(encoding="utf-8"))
    r = run(cell_key, cond_key, seed, taus=thr["thresholds"])
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(r) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def boot_ci(vals, n=2000, seed=0):
    """Seed-level bootstrap percentile interval for the mean."""
    if not vals:
        return None
    rs = np.random.RandomState(seed)
    a = np.asarray(vals, float)
    m = rs.randint(0, len(a), size=(n, len(a)))
    bs = a[m].mean(axis=1)
    return [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def mode_agg(eval_dir, thr_path, out_path):
    rows = []
    for p in sorted(Path(eval_dir).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no evaluation records in %s" % eval_dir)
    thr = json.loads(Path(thr_path).read_text(encoding="utf-8"))

    summary = {}
    for cell_key in CELLS:
        for cond in CONDITIONS:
            rs = [r for r in rows if r["cell"] == cell_key
                  and r["condition"] == cond]
            if not rs:
                continue
            seeds = sorted(r["seed"] for r in rs)
            adv = [r["adv_updates_admitted"] for r in rs]
            acc = [r["final_accuracy"] for r in rs]
            aucacc = [r["auc_accuracy"] for r in rs]
            hw = [r["honest_wrongly_isolated_rounds"] for r in rs]
            fp = [len(r["honest_agents_ever_flagged"]) for r in rs]
            ra = [len(r["returning_readmitted"]) for r in rs]
            # per-attacker detection, averaged over seeds
            det = {}
            for a in sorted(BASE_PROFILE) + [REVOKED]:
                v = [r["per_attacker_detection_rate"].get(str(a))
                     for r in rs if r["per_attacker_detection_rate"].get(str(a))
                     is not None]
                if v:
                    det[str(a)] = round(st.mean(v), 4)
            # flipper identified at all, per seed
            flip_found = [1 if r["per_attacker_detection_rate"].get(
                str(FLIPPER), 0) > 0 else 0 for r in rs]
            recon = [r["audit_reconstruction_rate"] for r in rs
                     if r["audit_reconstruction_rate"] is not None]
            summary["%s|%s" % (cell_key, cond)] = {
                "cell": cell_key, "condition": cond,
                "deployable": rs[0]["deployable"],
                "n_seeds": len(rs), "seeds": seeds,
                "adv_opportunities": rs[0]["adv_opportunities"],
                "adv_admitted_mean": round(st.mean(adv), 2),
                "adv_admitted_sd": round(st.pstdev(adv), 2),
                "adv_admitted_ci": boot_ci(adv),
                "adv_admitted_share": round(
                    st.mean(adv) / rs[0]["adv_opportunities"], 4),
                "final_accuracy_mean": round(st.mean(acc), 4),
                "final_accuracy_sd": round(st.pstdev(acc), 4),
                "final_accuracy_ci": boot_ci(acc),
                "auc_accuracy_mean": round(st.mean(aucacc), 4),
                "honest_wrong_rounds_mean": round(st.mean(hw), 2),
                "honest_wrong_rounds_ci": boot_ci(hw),
                "honest_agents_flagged_mean": round(st.mean(fp), 3),
                "returning_readmitted_mean": round(st.mean(ra), 2),
                "flipper_found_seeds": "%d of %d" % (sum(flip_found), len(rs)),
                "per_attacker_detection_rate": det,
                "audit_reconstruction_rate": (round(st.mean(recon), 4)
                                              if recon else None),
            }

    # paired comparisons against the deployable conventional baseline
    paired = {}
    for cell_key in CELLS:
        base = {r["seed"]: r for r in rows
                if r["cell"] == cell_key and r["condition"] == "conv_norm"}
        for cond in CONDITIONS:
            if cond == "conv_norm":
                continue
            other = {r["seed"]: r for r in rows
                     if r["cell"] == cell_key and r["condition"] == cond}
            common = sorted(set(base) & set(other))
            if len(common) < 3:
                continue
            d_adv = [other[s]["adv_updates_admitted"]
                     - base[s]["adv_updates_admitted"] for s in common]
            d_acc = [other[s]["final_accuracy"]
                     - base[s]["final_accuracy"] for s in common]
            try:
                from scipy.stats import wilcoxon
                nz = [v for v in d_adv if v != 0]
                p_adv = (round(float(wilcoxon(nz, mode="exact").pvalue), 5)
                         if 1 <= len(nz) <= 25 else
                         (round(float(wilcoxon(nz).pvalue), 5) if nz else None))
                nza = [v for v in d_acc if v != 0]
                p_acc = (round(float(wilcoxon(nza, mode="exact").pvalue), 5)
                         if 1 <= len(nza) <= 25 else
                         (round(float(wilcoxon(nza).pvalue), 5) if nza else None))
            except Exception as exc:
                p_adv = p_acc = "wilcoxon failed: %s" % type(exc).__name__
            paired["%s|%s vs conv_norm" % (cell_key, cond)] = {
                "n_pairs": len(common),
                "adv_admitted_diff_mean": round(st.mean(d_adv), 3),
                "adv_admitted_diff_ci": boot_ci(d_adv),
                "adv_admitted_wins": sum(1 for v in d_adv if v < 0),
                "adv_admitted_ties": sum(1 for v in d_adv if v == 0),
                "adv_admitted_losses": sum(1 for v in d_adv if v > 0),
                "adv_admitted_exact_p": p_adv,
                "accuracy_diff_mean": round(st.mean(d_acc), 4),
                "accuracy_diff_ci": boot_ci(d_acc),
                "accuracy_exact_p": p_acc,
            }

    Path(out_path).write_text(json.dumps({
        "experiment": "deployable online governance",
        "calibration_seeds": thr["calibration_seeds"],
        "evaluation_seeds": EVAL_SEEDS,
        "attack_start": ATTACK_START, "return_round": RETURN_ROUND,
        "returning": sorted(RETURNING), "revoked": REVOKED,
        "profile": BASE_PROFILE, "root_per_class": ROOT_PER_CLASS,
        "sdf03_enforced_fail_kl": SDF03_FAIL_KL,
        "frozen_thresholds": thr["thresholds"],
        "cells": CELLS,
        "conditions": {k: {"deployable": v["deployable"],
                           "channel": v["channel"],
                           "persist": v["persist"], "aggr": v["aggr"]}
                       for k, v in CONDITIONS.items()},
        "n_records": len(rows),
        "summary": summary, "paired_vs_conv_norm": paired,
    }, indent=1), encoding="utf-8")
    print("wrote", out_path, "from", len(rows), "records")
    return 0


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "calib":
        raise SystemExit(mode_calib(a[1], int(a[2]), a[3]))
    if a[0] == "freeze":
        raise SystemExit(mode_freeze(a[1], a[2]))
    if a[0] == "eval":
        raise SystemExit(mode_eval(a[1], a[2], int(a[3]), a[4], a[5]))
    if a[0] == "agg":
        raise SystemExit(mode_agg(a[1], a[2], a[3]))
    raise SystemExit("unknown mode %r" % a[0])
