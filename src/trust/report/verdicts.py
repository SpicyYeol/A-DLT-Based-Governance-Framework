# -*- coding: utf-8 -*-
"""The governance deliverable itself: the filled-in Structure Trust
Factors table, the per-agent verdicts and the machine-readable summary.

Split out of the single report module; the code is unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .. import config
from ..taxonomy import ALL_CONTROLS


def write_structure_trust_factors(results: dict, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "structure_trust_factors_filled.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["Level (3)", "Category (12)", "Sub-Category (36)",
                    "Detailed Category", "Control points", "Metric", "Threshold",
                    "Dimension", "Method", "Orientation",
                    "Status", "Score"])
        for cp in ALL_CONTROLS:
            r = results["controls"][cp.id]
            w.writerow([cp.level, cp.category, cp.subcategory,
                        cp.detailed_category, cp.control_point, cp.metric,
                        cp.threshold, cp.dimension, cp.method, cp.orient,
                        r["status"], round(r["score"], 4)])
    return path


def write_agent_verdicts(results: dict, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    path = out_dir / "agent_verdicts.csv"
    agents = results["verdict"]["agents"]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["agent_id", "principal", "behaviour", "is_adversary",
                    "trust_score", "verdict", "failed_controls"])
        for aid in sorted(agents, key=lambda x: int(x)):
            a = agents[aid]
            w.writerow([a["agent_id"], a["principal"], a["behaviour"],
                        a["is_adversary"], a["trust_score"], a["verdict"],
                        ";".join(f"{k}({v})" for k, v in a["failed_controls"].items())])
    return path


def write_summary_json(results: dict, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    path = out_dir / "governance_report.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return path
