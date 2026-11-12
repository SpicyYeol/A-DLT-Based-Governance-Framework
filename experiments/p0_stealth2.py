# -*- coding: utf-8 -*-
"""Same baseline audit as p0_baseline.py, under the stealth profile.

Fixes two errors in the first attempt: the revoked participant is still
active under this profile and therefore belongs in the ground-truth set,
and the behaviour labels must come from the stealth profile rather than
the primary one.

Usage: python p0_stealth2.py <out.json>
"""
from __future__ import annotations

import io
import json
import os
import statistics as st
import sys
from pathlib import Path

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                   # noqa: E402
from trust import aggregators as agg                        # noqa: E402
from trust import fl_simulation as fl                       # noqa: E402
from trust.engine import build_ctx, evaluate_controls, agent_verdict  # noqa: E402
from trust.taxonomy import ALL_CONTROLS                     # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
ALL_IDS = {cp.id for cp in ALL_CONTROLS}
CONV = {"M-CV-03", "S-MI-02", "S-DF-02", "L-IA-03", "M-CV-02"}
CONV_NORM = CONV | {"M-BC-01"}

PROFILE = config.STEALTH_PROFILE
TRUTH = set(PROFILE) | {config.REVOKED_AGENT}
LABEL = {a: PROFILE.get(a, "revoked cred." if a == config.REVOKED_AGENT
                        else "honest") for a in range(config.NUM_AGENTS)}

CONDITIONS = [
    ("none",        None,      None),
    ("ground_truth", "GT",     None),
    ("conv",        CONV,      None),
    ("conv+MK",     CONV,      agg.multi_krum),
    ("conv+TM",     CONV,      agg.trimmed_mean),
    ("convnorm",    CONV_NORM, None),
    ("convnorm+MK", CONV_NORM, agg.multi_krum),
    ("governance",  ALL_IDS,   None),
]


def flagged_with_reasons(res, ids, enabled):
    disabled = frozenset(ALL_IDS - enabled)
    out = {}
    for a in ids:
        verdict, failed = agent_verdict(res, a, disabled=disabled)
        if verdict != "Trusted":
            out[a] = sorted(failed)
    return out


def main(out_path):
    per_cond = {name: [] for name, _, _ in CONDITIONS}
    for s in SEEDS:
        prep = fl.prepare(s)
        rec = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                  profile=PROFILE)
        ctx = build_ctx(rec["signals"])
        res = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        for name, enabled, aggr in CONDITIONS:
            if enabled is None:
                excl, reasons = set(), {}
            elif enabled == "GT":
                excl, reasons = set(TRUTH), {}
            else:
                reasons = flagged_with_reasons(res, ids, enabled)
                excl = set(reasons)
            acc = fl.train_trajectory(prep, exclude=frozenset(excl),
                                      profile=PROFILE,
                                      aggregator=aggr)["accuracy"]
            tp = len(excl & TRUTH); fp = len(excl - TRUTH)
            fn = len(TRUTH - excl); tn = len(set(ids) - TRUTH - excl)
            per_cond[name].append({
                "seed": s, "accuracy": acc, "flagged": sorted(excl),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1) if (tp + fp) else 1.0,
                "fpr": fp / max(fp + tn, 1),
                "missed": sorted(TRUTH - excl),
                "reasons": {str(k): v for k, v in sorted(reasons.items())}})
        print("  seed %d done" % s, flush=True)

    lines = ["profile: %s" % PROFILE, "ground truth: %s" % sorted(TRUTH), "",
             "%-13s %17s %8s %8s %8s  %s"
             % ("condition", "accuracy", "recall", "prec", "FPR", "missed")]
    summary = {}
    for name, enabled, _ in CONDITIONS:
        rs = per_cond[name]
        a = [r["accuracy"] for r in rs]
        missed = sorted({LABEL[m] for r in rs for m in r["missed"]})
        summary[name] = {
            "accuracy_mean": round(st.mean(a), 4),
            "accuracy_sd": round(st.pstdev(a), 4),
            "recall": round(st.mean(r["recall"] for r in rs), 4),
            "precision": round(st.mean(r["precision"] for r in rs), 4),
            "fpr": round(st.mean(r["fpr"] for r in rs), 4),
            "missed_behaviours": missed, "per_seed": rs}
        d = summary[name]
        lines.append("%-13s %8.4f+-%.4f %8.3f %8.3f %8.3f  %s"
                     % (name, d["accuracy_mean"], d["accuracy_sd"], d["recall"],
                        d["precision"], d["fpr"], ", ".join(missed) or "-"))
    text = "\n".join(lines)
    print(); print(text)
    Path(out_path).write_text(json.dumps(
        {"profile": {str(k): v for k, v in PROFILE.items()},
         "ground_truth": sorted(TRUTH), "summary": summary}, indent=1),
        encoding="utf-8")
    io.open(out_path.replace(".json", ".txt"), "w",
            encoding="utf-8").write(text)
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
