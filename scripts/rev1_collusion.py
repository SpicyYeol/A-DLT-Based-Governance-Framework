# -*- coding: utf-8 -*-
"""Revision 1 (IEEE Access) — R2-2: empirical check of Proposition 2.

N = 8 participants. k = 1..6 identities collude by submitting identically
scaled updates (factor 12); agent 7 tampers with its payload (an
evidence-local violation) in every condition; no revoked credential.
Prediction: M-BC-01 (median/MAD) stops flagging the colluders once
k >= N/2, while the evidence-local tampering violation stays detected.
Run:  TRUST_DATA_ROOT=<dir> python scripts/rev1_collusion.py
"""
from __future__ import annotations
import json, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trust import config
from trust import fl_simulation as fl
from trust.engine import build_ctx, evaluate_controls
from trust.experiments import _conf, _flagged_set

OUT = config.DATA_ROOT / "reports_rev1"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
ORDER = [1, 3, 5, 2, 4, 6]
TAMPER = 7
config.REVOKED_AGENT = -1

rows = []
for k in range(1, 7):
    colluders = ORDER[:k]
    profile = {a: "scale_attack" for a in colluders}
    profile[TAMPER] = "tamper"
    honest = [a for a in range(8) if a not in profile]
    per = []
    for s in SEEDS:
        prep = fl.prepare(s)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
        ctx = build_ctx(naive["signals"])
        res = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        flagged = _flagged_set(res, ids)
        bc = res["M-BC-01"].per_agent
        gov = fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"]
        per.append({
            "seed": s,
            "mbc01_colluders_flagged": sum(bc[a] < 0.5 for a in colluders),
            "colluders_flagged": sum(a in flagged for a in colluders),
            "tamper_flagged": TAMPER in flagged,
            "honest_flagged": sum(a in flagged for a in honest),
            "naive": naive["accuracy"], "gov": gov})
    rows.append({
        "k": k, "fraction": k / 8, "n_honest": len(honest),
        "colluder_recall": sum(p["colluders_flagged"] for p in per) / (k * len(SEEDS)),
        "mbc01_recall": sum(p["mbc01_colluders_flagged"] for p in per) / (k * len(SEEDS)),
        "tamper_recall": sum(p["tamper_flagged"] for p in per) / len(SEEDS),
        "honest_fp": sum(p["honest_flagged"] for p in per),
        "honest_n": len(honest) * len(SEEDS),
        "naive": st.mean(p["naive"] for p in per), "gov": st.mean(p["gov"] for p in per),
        "per_seed": per})
    print(json.dumps({k_: v for k_, v in rows[-1].items() if k_ != "per_seed"}), flush=True)
json.dump(rows, open(OUT / "rev1_collusion.json", "w"), indent=1)
