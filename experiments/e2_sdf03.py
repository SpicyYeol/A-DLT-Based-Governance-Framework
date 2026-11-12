# -*- coding: utf-8 -*-
"""E2: four ways of supplying the label-distribution signal for S-DF-03.

  oracle          the true local label distribution (ideal reference only)
  gradient        recovered from the submitted update (our earlier attempt)
  server_anchored a small trusted server set, FLTrust-style direction score
  conv_norm       no label-distribution control at all (the fair baseline)

Cells: label-flip strength {0.30, 0.70} x partition {near-IID, Dirichlet 0.5}.
Thresholds are fitted on the calibration seeds and FROZEN before the
evaluation seeds are touched. The evaluation seeds have not been used by any
earlier experiment.

Usage: python e2_sdf03.py <out.json> [n_calib] [n_eval]
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
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp12"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                        # noqa: E402
from trust import fl_simulation as fl                            # noqa: E402
from trust import fltrust                                        # noqa: E402
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402
from trust.ledger import Ledger, sha256_hex, sign                 # noqa: E402
from trust.taxonomy import ALL_CONTROLS                           # noqa: E402

ALL_IDS = {cp.id for cp in ALL_CONTROLS}
CONV_NORM = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02", "M-BC-01"}
PROFILE = config.ADVERSARY_PROFILE
TRUTH = set(PROFILE) | {config.REVOKED_AGENT}
FLIPPER = 1
EPS = 1e-9

CALIB_START = 102          # unused by every earlier experiment
EVAL_START = 110

CELLS = [(0.30, "iid", 1.0), (0.70, "iid", 1.0),
         (0.30, "dirichlet", 0.5), (0.70, "dirichlet", 0.5)]


def kl(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
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
    tot = est.sum()
    return est / tot if tot > 0 else np.asarray(prior, float)


def trajectory(prep, seed, flip_rate):
    """One trajectory with a parameterised label-flip rate; mirrors
    fl_simulation.train_trajectory so results stay comparable."""
    rng = np.random.RandomState(seed)
    backbone = prep.get("backbone", "linear")
    X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]
    shards, base_dist = prep["shards"], prep["base_dist"]
    classes = np.unique(y_tr)

    def local_fit(gW, Xl, yl):
        m = fl._new_model(backbone, seed)
        m.partial_fit(Xl[:256], yl[:256], classes=classes)
        fl._set_w(m, gW[0], gW[1])
        for _ in range(fl.EPOCHS):
            o = rng.permutation(len(Xl))
            m.partial_fit(Xl[o], yl[o])
        return m

    def wavg(Ws, ws):
        w = np.asarray(ws, float); w /= w.sum()
        return ([sum(w[k] * Ws[k][0][l] for k in range(len(Ws)))
                 for l in range(len(Ws[0][0]))],
                [sum(w[k] * Ws[k][1][l] for k in range(len(Ws)))
                 for l in range(len(Ws[0][1]))])

    root = []
    for c in range(prep["n_classes"]):
        root.extend(np.where(y_tr == c)[0][:10].tolist())
    X_root, y_root = X_tr[np.array(root)], y_tr[np.array(root)]

    ledger = Ledger(); ledger.reset()
    gm = fl._new_model(backbone, seed)
    gm.partial_fit(X_tr[:256], y_tr[:256], classes=classes)

    agents = {sh["agent_id"]: {
        "agent_id": sh["agent_id"], "principal": fl._principal(sh["agent_id"]),
        "behaviour": PROFILE.get(sh["agent_id"], "honest"),
        "is_adversary": sh["agent_id"] in PROFILE,
        "n_samples": int(len(sh["y"])), "rounds": []} for sh in shards}
    upd = {sh["agent_id"]: [] for sh in shards}
    srv = []
    now = 1_700_000_000

    for rnd in range(config.NUM_ROUNDS):
        gW = fl._get_w(gm); contribs = []
        ms = local_fit(gW, X_root, y_root); sW = fl._get_w(ms)
        srv.append(np.concatenate(
            [(a - b).ravel() for a, b in zip(sW[0], gW[0])]
            + [(a - b).ravel() for a, b in zip(sW[1], gW[1])]))

        for sh in shards:
            aid = sh["agent_id"]
            beh = PROFILE.get(aid, "honest")
            Xl, yl = sh["X"].copy(), sh["y"].copy()
            if beh == "label_flip":
                yl = yl.copy()
                yl[rng.rand(len(yl)) < flip_rate] = 0
            m = local_fit(gW, Xl, yl); lW = fl._get_w(m)
            d_c = [a - b for a, b in zip(lW[0], gW[0])]
            d_i = [a - b for a, b in zip(lW[1], gW[1])]
            if beh == "scale_attack":
                d_c = [x * 12.0 for x in d_c]; d_i = [x * 12.0 for x in d_i]
            new_c = [g + d for g, d in zip(gW[0], d_c)]
            new_i = [g + d for g, d in zip(gW[1], d_i)]
            upd[aid].append(np.concatenate([a.ravel() for a in d_c]
                                           + [a.ravel() for a in d_i]))
            contribs.append({"W": (new_c, new_i), "weight": len(yl), "aid": aid,
                             "coef": new_c[0], "int": new_i[0]})
            un = float(sum(np.linalg.norm(a) for a in d_c)
                       + sum(np.linalg.norm(a) for a in d_i))
            klv = fl._label_kl(yl, prep["n_classes"], base_dist)
            payload = ("agent%d-round%d-n%d" % (aid, rnd, len(yl))).encode()
            ch = sg = None; ts = now
            if beh == "stale":
                ts = now - config.FRESHNESS_SLA_SEC * 5
            if beh == "tamper":
                ch = sha256_hex(payload + b"-tampered")
                sg = sign(aid, sha256_hex(payload))
            blk = ledger.append(agent_id=aid, principal=fl._principal(aid),
                                tx_type="data_packet", payload=payload,
                                claimed_hash=ch, signature=sg, timestamp=ts,
                                ingest_time=now,
                                meta={"round": rnd, "behaviour": beh,
                                      "local_acc": 0.0,
                                      "update_norm": round(un, 6),
                                      "label_kl": round(klv, 6),
                                      "logged": True, "model_version": rnd + 1})
            agents[aid]["rounds"].append({
                "round": rnd, "update_norm": un, "local_acc": 0.0,
                "label_kl": klv, "block_index": blk.index, "timestamp": ts,
                "ingest_time": now, "model_version": rnd + 1})
        cc, ii = wavg([c["W"] for c in contribs], [c["weight"] for c in contribs])
        fl._set_w(gm, cc, ii)

    signals = {"dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
               "num_rounds": config.NUM_ROUNDS, "global_accuracy": 0.0,
               "revoked_agent": config.REVOKED_AGENT,
               "signing_scheme": config.SIGNING_SCHEME, "agents": agents,
               "seed": seed, "generated_at": now}
    return {"signals": signals,
            "updates": {a: np.mean(v, axis=0) for a, v in upd.items()},
            "server_update": np.mean(srv, axis=0),
            "prior": np.asarray(prep["base_dist"], float),
            "n_classes": prep["n_classes"],
            "contrib_fn": wavg, "prep": prep, "seed": seed}


def scores_for(tr):
    """Per-agent score from each channel (higher = more suspicious)."""
    sig = tr["signals"]; ids = sorted(sig["agents"])
    ncls = tr["n_classes"]; prior = tr["prior"]
    oracle = {i: float(np.mean([r["label_kl"]
                                for r in sig["agents"][i]["rounds"]]))
              for i in ids}
    grad = {i: kl(estimate_dist(np.asarray(tr["updates"][i])[-ncls:], prior),
                  prior) for i in ids}
    ft = fltrust.fltrust_scores({i: tr["updates"][i] for i in ids},
                                tr["server_update"])
    anchored = {i: 1.0 - ft[i]["relu_cos"] for i in ids}   # higher = worse
    return {"oracle": oracle, "gradient": grad, "server_anchored": anchored}


def auroc(scores, positives):
    pos = [v for k, v in scores.items() if k in positives]
    neg = [v for k, v in scores.items() if k not in positives]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def pick_threshold(vals_pos, vals_neg):
    """Midpoint if separable; otherwise the cut maximising Youden's J."""
    if vals_pos and vals_neg and min(vals_pos) > max(vals_neg):
        return (min(vals_pos) + max(vals_neg)) / 2.0
    cand = sorted(set(vals_pos + vals_neg))
    best, best_j = (cand[0] if cand else 0.0), -2.0
    for c in cand:
        tpr = sum(1 for v in vals_pos if v > c) / max(len(vals_pos), 1)
        fpr = sum(1 for v in vals_neg if v > c) / max(len(vals_neg), 1)
        if tpr - fpr > best_j:
            best_j, best = tpr - fpr, c
    return best


def evaluate(trs, taus):
    """Apply frozen thresholds; return metrics per channel."""
    out = {}
    for ch in ("oracle", "gradient", "server_anchored", "conv_norm"):
        tp = fp = fn = tn = 0
        flip_hits = 0
        accs = []
        auc = []
        for tr in trs:
            sig = tr["signals"]; ids = sorted(sig["agents"])
            ctx = build_ctx(sig)
            res = evaluate_controls(ctx)
            base = set()
            for a in ids:
                if agent_verdict(res, a,
                                 disabled=frozenset(ALL_IDS - CONV_NORM))[0] != "Trusted":
                    base.add(a)
            if ch == "conv_norm":
                flagged = set(base)
            else:
                sc = tr["_scores"][ch]
                flagged = base | {i for i in ids if sc[i] > taus[ch]}
                auc.append(auroc(sc, {FLIPPER}))
            if FLIPPER in flagged:
                flip_hits += 1
            tp += len(flagged & TRUTH); fp += len(flagged - TRUTH)
            fn += len(TRUTH - flagged); tn += len(set(ids) - TRUTH - flagged)
            acc = fl.train_trajectory(tr["prep"], exclude=frozenset(flagged),
                                      profile=PROFILE)["accuracy"]
            accs.append(acc)
        out[ch] = {
            "flipper_recall": round(flip_hits / len(trs), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4) if (tp + fp) else 1.0,
            "honest_fpr": round(fp / max(fp + tn, 1), 4),
            "auroc_flipper": (round(st.mean(auc), 4) if auc else None),
            "accuracy": round(st.mean(accs), 4),
            "accuracy_sd": round(st.pstdev(accs), 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return out


def main(out_path, n_cal=5, n_eval=20):
    report = {}
    for flip, part, alpha in CELLS:
        key = "flip%.0f_%s" % (flip * 100, part)
        print("== cell %s" % key, flush=True)
        cal_trs, ev_trs = [], []
        for k in range(n_cal):
            s = CALIB_START + k
            prep = fl.prepare(s, partition=part, alpha=alpha)
            tr = trajectory(prep, s, flip); tr["_scores"] = scores_for(tr)
            cal_trs.append(tr)
        taus = {}
        for ch in ("gradient", "server_anchored"):
            pos = [t["_scores"][ch][FLIPPER] for t in cal_trs]
            neg = [v for t in cal_trs for i, v in t["_scores"][ch].items()
                   if i != FLIPPER]
            taus[ch] = round(pick_threshold(pos, neg), 4)
        pos = [t["_scores"]["oracle"][FLIPPER] for t in cal_trs]
        neg = [v for t in cal_trs for i, v in t["_scores"]["oracle"].items()
               if i != FLIPPER]
        taus["oracle"] = 0.20                      # the published rule
        print("   thresholds (frozen): %s" % taus, flush=True)

        for k in range(n_eval):
            s = EVAL_START + k
            prep = fl.prepare(s, partition=part, alpha=alpha)
            tr = trajectory(prep, s, flip); tr["_scores"] = scores_for(tr)
            ev_trs.append(tr)

        report[key] = {"flip_rate": flip, "partition": part, "alpha": alpha,
                       "thresholds": taus,
                       "calibration_seeds": [CALIB_START + k for k in range(n_cal)],
                       "evaluation_seeds": [EVAL_START + k for k in range(n_eval)],
                       "calibration": evaluate(cal_trs, taus),
                       "evaluation": evaluate(ev_trs, taus)}
        e = report[key]["evaluation"]
        print("   %-16s %8s %8s %8s %8s %8s" % ("channel", "flipRec", "recall",
                                                "FPR", "AUROC", "acc"))
        for ch in ("oracle", "gradient", "server_anchored", "conv_norm"):
            d = e[ch]
            print("   %-16s %8.3f %8.3f %8.3f %8s %8.4f"
                  % (ch, d["flipper_recall"], d["recall"], d["honest_fpr"],
                     ("%.3f" % d["auroc_flipper"]) if d["auroc_flipper"] else "-",
                     d["accuracy"]), flush=True)

    Path(out_path).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(a[0], int(a[1]) if len(a) > 1 else 5,
                          int(a[2]) if len(a) > 2 else 20))
