# -*- coding: utf-8 -*-
"""The three headline experiment tables: closed-loop gating, detection and
the ablation harm table.

Split out of the single report module; the code is unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .. import config
from ..taxonomy import ALL_CONTROLS


def write_experiment_csvs(exp: dict, out_dir: Path | None = None) -> dict:
    """Write the paper-ready experiment tables (E1 closed-loop, E2 detection,
    E3 ablation). Returns a dict of written paths."""
    out_dir = Path(out_dir or config.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # E1 — closed-loop gating, one row per seed + an aggregate row
    p1 = out_dir / "exp_closed_loop.csv"
    with p1.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "naive_accuracy", "governance_accuracy",
                    "oracle_accuracy", "gating_uplift", "detection_recall",
                    "detection_precision", "detection_f1", "false_positive_rate"])
        for r in exp["per_seed"]:
            d = r["detection"]
            w.writerow([r["seed"], r["naive_accuracy"], r["governance_accuracy"],
                        r["oracle_accuracy"], r["gating_uplift"], d["recall"],
                        d["precision"], d["f1"], d["false_positive_rate"]])
        e1 = exp["summary"]["E1_closed_loop"]
        e2 = exp["summary"]["E2_detection"]
        w.writerow([f"mean±std (n={e1['naive_accuracy']['n']})",
                    f"{e1['naive_accuracy']['mean']}±{e1['naive_accuracy']['std']}",
                    f"{e1['governance_accuracy']['mean']}±{e1['governance_accuracy']['std']}",
                    f"{e1['oracle_accuracy']['mean']}±{e1['oracle_accuracy']['std']}",
                    f"{e1['gating_uplift']['mean']}±{e1['gating_uplift']['std']}",
                    f"{e2['recall']['mean']}±{e2['recall']['std']}",
                    f"{e2['precision']['mean']}±{e2['precision']['std']}",
                    f"{e2['f1']['mean']}±{e2['f1']['std']}",
                    f"{e2['false_positive_rate']['mean']}±{e2['false_positive_rate']['std']}"])
    paths["closed_loop"] = str(p1)

    # E2 — detection confusion matrix (from the first seed for the raw counts)
    p2 = out_dir / "exp_detection.csv"
    with p2.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "mean", "std", "n"])
        for k in ("precision", "recall", "f1", "false_positive_rate"):
            m = exp["summary"]["E2_detection"][k]
            w.writerow([k, m["mean"], m["std"], m["n"]])
        # confusion counts per seed
        w.writerow([])
        w.writerow(["seed", "TP", "FP", "FN", "TN"])
        for r in exp["per_seed"]:
            d = r["detection"]
            w.writerow([r["seed"], d["TP"], d["FP"], d["FN"], d["TN"]])
    paths["detection"] = str(p2)

    # E3 — ablation harm table
    p3 = out_dir / "exp_ablation.csv"
    with p3.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["ablation", "risk", "primary_harm",
                    "accuracy_under_ablation(mean±std)", "delta_vs_governance",
                    "n_undetected_adversaries(mean±std)"])
        for key, v in exp["summary"]["E3_ablation"].items():
            acc, dl, und = (v["accuracy_under_ablation"], v["delta_vs_governance"],
                            v["n_undetected_adversaries"])
            w.writerow([v["name"], v["risk"], v["primary_harm"],
                        f"{acc['mean']}±{acc['std']}",
                        f"{dl['mean']}±{dl['std']}",
                        f"{und['mean']}±{und['std']}"])
    paths["ablation"] = str(p3)
    return paths
