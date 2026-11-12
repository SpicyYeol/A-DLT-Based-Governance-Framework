# -*- coding: utf-8 -*-
"""P0: audit the conventional-checks baseline and rebuild it fairly.

Prints, for every condition and seed, which participant was flagged by which
control, so the recall figure can be traced to a detection event instead of
being asserted.

Conditions
  none         no defense (FedAvg over all 8)
  ground_truth the injected set excluded
  conv         conventional security checks: signature verification (M-CV-03),
               payload hash (S-MI-02), freshness/replay (S-DF-02), credential
               revocation and expiry (L-IA-03, M-CV-02)
  conv+MK      conv, aggregated with Multi-Krum
  conv+TM      conv, aggregated with Trimmed-Mean
  convnorm     conv plus an update-norm bound (M-BC-01) -- the strongest
               conventional baseline a real deployment would run
  convnorm+MK  convnorm, aggregated with Multi-Krum
  governance   the full 36-control framework

Usage: python p0_baseline.py <out.json>
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

BEHAV = {0: "honest", 1: "label flipping", 2: "honest", 3: "scaling",
         4: "honest", 5: "replay/stale", 6: "tampering", 7: "revoked cred."}

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
    """Return {aid: [control ids that failed within `enabled`]}."""
    disabled = frozenset(ALL_IDS - enabled)
    out = {}
    for a in ids:
        verdict, failed = agent_verdict(res, a, disabled=disabled)
        if verdict != "Trusted":
            out[a] = sorted(failed)
    return out


def main(out_path):
    profile = config.ADVERSARY_PROFILE
    truth = set(profile) | {config.REVOKED_AGENT}
    log = []
    per_cond = {name: [] for name, _, _ in CONDITIONS}

    for s in SEEDS:
        prep = fl.prepare(s)
        rec = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                  profile=profile)
        ctx = build_ctx(rec["signals"])
        res = evaluate_controls(ctx)
        ids = ctx.agent_ids()

        for name, enabled, aggr in CONDITIONS:
            if enabled is None:
                excl, reasons = set(), {}
            elif enabled == "GT":
                excl, reasons = set(truth), {}
            else:
                reasons = flagged_with_reasons(res, ids, enabled)
                excl = set(reasons)
            acc = fl.train_trajectory(prep, exclude=frozenset(excl),
                                      profile=profile,
                                      aggregator=aggr)["accuracy"]
            tp = len(excl & truth)
            fp = len(excl - truth)
            fn = len(truth - excl)
            tn = len(set(ids) - truth - excl)
            per_cond[name].append({
                "seed": s, "accuracy": acc, "flagged": sorted(excl),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1) if (tp + fp) else 1.0,
                "fpr": fp / max(fp + tn, 1),
                "missed": sorted(truth - excl),
                "reasons": {str(k): v for k, v in sorted(reasons.items())},
            })
        log.append("seed %d: per-agent failures = %s" % (
            s, {a: sorted(agent_verdict(res, a)[1]) for a in ids}))
        print("  seed %d done" % s, flush=True)

    lines = []
    lines.append("%-13s %17s %8s %8s %8s  %s"
                 % ("condition", "accuracy", "recall", "prec", "FPR", "missed"))
    summary = {}
    for name, enabled, _ in CONDITIONS:
        rs = per_cond[name]
        a = [r["accuracy"] for r in rs]
        missed = sorted({BEHAV[m] for r in rs for m in r["missed"]})
        summary[name] = {
            "enabled_controls": (sorted(enabled) if isinstance(enabled, set)
                                 else enabled),
            "accuracy_mean": round(st.mean(a), 4),
            "accuracy_sd": round(st.pstdev(a), 4),
            "recall": round(st.mean(r["recall"] for r in rs), 4),
            "precision": round(st.mean(r["precision"] for r in rs), 4),
            "fpr": round(st.mean(r["fpr"] for r in rs), 4),
            "missed_behaviours": missed,
            "per_seed": rs,
        }
        d = summary[name]
        lines.append("%-13s %8.4f+-%.4f %8.3f %8.3f %8.3f  %s"
                     % (name, d["accuracy_mean"], d["accuracy_sd"],
                        d["recall"], d["precision"], d["fpr"],
                        ", ".join(missed) or "-"))

    lines.append("")
    lines.append("detection reasons, seed %d:" % SEEDS[0])
    for name, enabled, _ in CONDITIONS:
        if not isinstance(enabled, set):
            continue
        r0 = summary[name]["per_seed"][0]["reasons"]
        lines.append("  %-13s %s" % (name, {("%s %s" % (k, BEHAV[int(k)])): v
                                            for k, v in r0.items()}))
    text = "\n".join(lines)
    print()
    print(text)

    Path(out_path).write_text(json.dumps(
        {"seeds": SEEDS, "conventional_controls": sorted(CONV),
         "conventional_plus_norm": sorted(CONV_NORM),
         "summary": summary}, indent=1), encoding="utf-8")
    io.open(out_path.replace(".json", ".txt"), "w",
            encoding="utf-8").write(text + "\n\n" + "\n".join(log))
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
