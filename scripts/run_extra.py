# -*- coding: utf-8 -*-
"""Two supplementary measurements, printed for transcription into the paper.

  (1) combined baseline: governance isolation, then robust aggregation over
      the surviving contributions
  (2) attestation band re-scoring at {1.0, 0.5, 0.0}, and its effect on the
      foundation score and on reputation

  Usage:  python scripts/run_extra.py
"""
from __future__ import annotations
import sys, os, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trust import config
from trust import fl_simulation as fl
from trust import aggregators as agg
from trust import attestations as att
from trust.engine import build_ctx, evaluate_controls, score
from trust.experiments import _flagged_set, detection_metrics

SEEDS = config.exp_seeds()          # [42..46]
MAIN = config.ADVERSARY_PROFILE
STEALTH = config.STEALTH_PROFILE


def agg_ms(xs):
    m = st.mean(xs); s = st.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.4f}±{s:.4f}"


# ══════════════════════════════════════════════════════════════════════
# (1) combined baseline
# ══════════════════════════════════════════════════════════════════════
def combined_baseline(profile, label):
    ROBUST = {"Multi-Krum": agg.multi_krum, "Trimmed-Mean": agg.trimmed_mean,
              "Coord-Median": agg.coordinate_median}
    rows = {k: [] for k in
            ["Governance+FedAvg", "Governance+Multi-Krum",
             "Governance+Trimmed-Mean", "Governance+Coord-Median",
             "Multi-Krum(alone)", "Trimmed-Mean(alone)"]}
    recalls, nsurv = [], []
    for seed in SEEDS:
        prep = fl.prepare(seed)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
        ctx = build_ctx(naive["signals"]); results = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        gt = set(profile) | {config.REVOKED_AGENT}
        flagged = _flagged_set(results, ids)
        recalls.append(detection_metrics(results, ids, gt)["recall"])
        nsurv.append(len(ids) - len(flagged))

        rows["Governance+FedAvg"].append(
            fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"])
        rows["Governance+Multi-Krum"].append(
            fl.train_trajectory(prep, exclude=flagged, profile=profile, aggregator=agg.multi_krum)["accuracy"])
        rows["Governance+Trimmed-Mean"].append(
            fl.train_trajectory(prep, exclude=flagged, profile=profile, aggregator=agg.trimmed_mean)["accuracy"])
        rows["Governance+Coord-Median"].append(
            fl.train_trajectory(prep, exclude=flagged, profile=profile, aggregator=agg.coordinate_median)["accuracy"])
        rows["Multi-Krum(alone)"].append(
            fl.train_trajectory(prep, profile=profile, aggregator=agg.multi_krum)["accuracy"])
        rows["Trimmed-Mean(alone)"].append(
            fl.train_trajectory(prep, profile=profile, aggregator=agg.trimmed_mean)["accuracy"])

    print(f"\n### (1) combined baseline - {label} profile  (n={len(SEEDS)} seeds)")
    print(f"  governance recall={agg_ms(recalls)}  survivors={agg_ms([float(x) for x in nsurv])}")
    print(f"  {'configuration':28s} accuracy (mean +/- SD)")
    for k, v in rows.items():
        print(f"  {k:28s} {agg_ms(v)}")
    return rows


# ══════════════════════════════════════════════════════════════════════
# (2) attestation band re-scoring at {1.0, 0.5, 0.0}
# ══════════════════════════════════════════════════════════════════════
# mapping rule (editable): structural or complete evidence = 1.0,
# declarative or partial evidence = 0.5, no evidence = 0.0
NEW_BAND = {
    "L-IA-02": 1.0,   # principal registry - structural
    "L-CI-02": 1.0,   # aggregator-only append - structural
    "L-GA-01": 1.0,   # all 36 controls documented - complete
    "L-GA-03": 0.5,   # ISO mapping kept in a README - declarative, partial
    "L-CT-01": 1.0,   # HMAC key pinned - structural
    "L-CT-03": 0.5,   # no live peer domain - partial
    "S-MI-03": 1.0,   # intra-process integrity - structural
    "S-AL-02": 1.0,   # append-only JSONL - complete
}


def reputation_snapshot(profile, seed):
    """Per-agent reputation and foundation score for one seed, using the
    ATTESTATIONS table as it currently stands."""
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    ctx = build_ctx(naive["signals"]); results = evaluate_controls(ctx)
    sc = score(ctx, results)
    agents = sc["agents"] if "agents" in sc else sc.get("per_agent", {})
    # score() returns a dict keyed by agent
    return sc


def attestation_rescore():
    print("\n### (2) attestation band re-scoring  {1.0, 0.5, 0.0}")
    print("  mapping (editable):", {k: NEW_BAND[k] for k in NEW_BAND})
    old = dict(att.ATTESTATIONS)   # keep the original to restore afterwards
    seed = config.RANDOM_SEED      # same condition as the reported value (42)

    # --- OLD ---
    sc_old = reputation_snapshot(MAIN, seed)
    # --- with the new band applied ---
    for cid, s in NEW_BAND.items():
        att.ATTESTATIONS[cid] = (s, att.ATTESTATIONS[cid][1])
    sc_new = reputation_snapshot(MAIN, seed)
    att.ATTESTATIONS.clear(); att.ATTESTATIONS.update(old)   # restore

    fo, fn = sc_old["foundation_score"], sc_new["foundation_score"]
    print(f"\n  foundation:  OLD={fo:.4f}  →  NEW={fn:.4f}  (Δ={fn-fo:+.4f})")
    print(f"\n  {'agent':5s} {'behaviour':12s} {'verdict':13s} {'rep_OLD':8s} {'rep_NEW':8s}  Δ")
    ao, an = sc_old["agents"], sc_new["agents"]
    for aid in sorted(ao):
        o, n = ao[aid], an[aid]
        print(f"  {aid:<5d} {str(o['behaviour']):12s} {o['verdict']:13s} "
              f"{o['trust_score']:.4f}  {n['trust_score']:.4f}  {n['trust_score']-o['trust_score']:+.4f}"
              + ("" if o['verdict'] == n['verdict'] else "  <verdict changed>"))
    # mean reputation over the five seeds as well
    print(f"\n  [mean reputation over five seeds]")
    reps_old = {i: [] for i in range(config.NUM_AGENTS)}
    reps_new = {i: [] for i in range(config.NUM_AGENTS)}
    for s in SEEDS:
        so = reputation_snapshot(MAIN, s)
        for cid, v in NEW_BAND.items():
            att.ATTESTATIONS[cid] = (v, att.ATTESTATIONS[cid][1])
        sn = reputation_snapshot(MAIN, s)
        att.ATTESTATIONS.clear(); att.ATTESTATIONS.update(old)
        for aid in so["agents"]:
            reps_old[aid].append(so["agents"][aid]["trust_score"])
            reps_new[aid].append(sn["agents"][aid]["trust_score"])
    print(f"  {'agent':5s} {'rep_OLD(5s)':16s} {'rep_NEW(5s)':16s}")
    for aid in sorted(reps_old):
        if reps_old[aid]:
            print(f"  {aid:<5d} {agg_ms(reps_old[aid]):16s} {agg_ms(reps_new[aid]):16s}")


def main() -> int:
    print("=" * 70)
    combined_baseline(MAIN, "main")
    combined_baseline(STEALTH, "stealth")
    attestation_rescore()
    print("\n" + "=" * 70 + "\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
