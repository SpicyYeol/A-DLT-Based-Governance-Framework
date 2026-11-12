# -*- coding: utf-8 -*-
"""Re-derive the paper's reported values from the deposited run outputs.

This script recomputes, from the per-seed records in results/, the numbers the
paper prints in its tables, so a reader can compare them against the article
without re-running any experiment. It asserts nothing about the article's text:
it prints what the deposited data gives and names the table each value belongs
to.

Standard deviations use the population estimator over the evaluated seeds,
dividing by n rather than n-1, which is the convention the paper states and
uses in every table.

    python verify/reproduce_tables.py

Exits non-zero only if a result file is missing or unreadable.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "results" / "local"
HOST = ROOT / "results" / "compute_host"

missing: list[str] = []


def load(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        missing.append("%s (%s)" % (path.name, exc.strerror))
        return None


def mean_sd(values) -> str:
    if len(values) < 2:
        return "%.4f" % st.fmean(values)
    return "%.4f +/- %.4f" % (st.fmean(values), st.pstdev(values))


def head(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


# --------------------------------------------------- Table 8: contributions
def table8() -> None:
    d = load(LOCAL / "mac5_out.json")
    if not d:
        return
    head("Table 8  marginal accuracy contribution, five seeds (mac5.py)")
    print("  %-22s %s" % ("participant", "MAC(a)"))
    for row in d["rows"]:
        label = "%d %s" % (row["agent"], row["behaviour"])
        print("  %-22s %s" % (label, mean_sd(row["per_seed"])))


# ------------------------------------- Table 10: layered baseline comparison
def table10() -> None:
    for name, profile in (("p0_out.json", "primary"),
                          ("p0_stealth2_out.json", "stealth")):
        d = load(LOCAL / name)
        if not d:
            continue
        head("Table 10  %s profile (%s)" % (profile, name))
        print("  %-16s %-22s %-8s %s" % ("condition", "accuracy", "recall", "honest FPR"))
        for cond, v in d["summary"].items():
            if not isinstance(v, dict) or "per_seed" not in v:
                continue
            acc = [x["accuracy"] for x in v["per_seed"]]
            rec = [x["recall"] for x in v["per_seed"]]
            fpr = [x["fpr"] for x in v["per_seed"]]
            print("  %-16s %-22s %-8.3f %.3f"
                  % (cond, mean_sd(acc), st.fmean(rec), st.fmean(fpr)))


# ----------------------------------------- Table 15: detection by family
def table15() -> None:
    d = load(LOCAL / "exp12_out.json")
    if not d:
        return
    head("Table 15  detection by control family, five seeds (exp12.py)")
    fam = [("All controls", "governance_full"),
           ("Deterministic only", "deterministic"),
           ("Statistical only", "statistical")]
    n_agents = 8
    for pname, prof in d["profiles"].items():
        print("  %s" % pname)
        for label, key in fam:
            prec, rec, fpr = [], [], []
            for row in prof["per_seed"]:
                bad = set(row["gt_bad"])
                honest = set(range(n_agents)) - bad
                flagged = set(row["flagged"][key])
                tp = len(flagged & bad)
                fp = len(flagged & honest)
                prec.append(tp / (tp + fp) if (tp + fp) else 1.0)
                rec.append(tp / len(bad) if bad else 0.0)
                fpr.append(fp / len(honest) if honest else 0.0)
            print("    %-20s precision %-18s recall %-18s honest FPR %s"
                  % (label, mean_sd(prec), mean_sd(rec), mean_sd(fpr)))


# ------------------------------- compute-host experiments: summary figures
def host_summaries() -> None:
    files = [("e4_summary.json", "Table 12  deployable evidence channels"),
             ("e5_summary.json", "Table 13  enforcement policies"),
             ("e6_summary.json", "Table 20  ledger ablation"),
             ("e6b_summary.json", "Table 21  auditability, four records"),
             ("e7_summary.json", "Table 19  participant count")]
    for name, title in files:
        d = load(HOST / name)
        if not d:
            continue
        head("%s (%s)" % (title, name))
        keys = [k for k in d if k not in ("per_seed", "rows")]
        for k in keys[:6]:
            v = d[k]
            text = json.dumps(v) if not isinstance(v, (str, int, float)) else str(v)
            print("  %-26s %s" % (k, text[:110]))
        print("  (full record: results/compute_host/%s;" % name)
        print("   per-seed rows: results/compute_host/e4e7_perseed.tar.gz)")


def main() -> int:
    print("Re-deriving reported values from the deposited run outputs")
    print("Standard deviations use the population estimator over the evaluated seeds.")
    table8()
    table10()
    table15()
    host_summaries()
    if missing:
        print()
        print("could not read %d file(s): %s" % (len(missing), ", ".join(missing)))
        return 1
    print()
    print("Done. Compare these against the tables in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
