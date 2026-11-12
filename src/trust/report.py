"""Report writers.

Produces the deliverable that mirrors the deck's slide 16/17 table — the
*filled-in* Structure Trust Factors with the previously-empty
``Detailed Category`` and ``Control points`` columns populated, plus the live
status/score for each of the 36 sub-categories.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import config
from .taxonomy import ALL_CONTROLS


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
        w.writerow(["ablation", "slide4_risk", "primary_harm",
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


def write_baseline_csv(bl: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_baselines.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["aggregation_rule", "global_accuracy(mean±std)", "n_seeds",
                    "catches_non_update_threats"])
        robust = set(("FedAvg", "Coord-Median", "Trimmed-Mean", "Krum", "Multi-Krum"))
        for name, m in bl["summary_accuracy"].items():
            catches = "no" if name in robust else \
                ("yes (identity/freshness/integrity)" if name == "Governance (ours)"
                 else "n/a (uses ground truth)")
            w.writerow([name, f"{m['mean']}±{m['std']}", m["n"], catches])
        d = bl["detection_ours"]
        w.writerow([])
        w.writerow(["governance detection", f"recall={d['recall']['mean']}",
                    f"precision={d['precision']['mean']}", f"F1={d['f1']['mean']}"])
    return str(p)


def write_stealth_csv(st: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_stealth.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        e1, e2 = st["E1_closed_loop"], st["E2_detection"]
        w.writerow(["profile", st["profile"]])
        for k in ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
                  "gating_uplift"):
            w.writerow([k, f"{e1[k]['mean']}±{e1[k]['std']}"])
        for k in ("precision", "recall", "f1", "false_positive_rate"):
            w.writerow([f"detection_{k}", f"{e2[k]['mean']}±{e2[k]['std']}"])
        w.writerow([])
        w.writerow(["injected_behaviour", "detection_rate", "detection_std"])
        std = st.get("per_behaviour_detection_std", {})
        for beh, rate in st["per_behaviour_detection_rate"].items():
            w.writerow([beh, rate, std.get(beh, 0.0)])
    return str(p)


def write_subcategory_capability_csv(sc: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_subcategory_capability.csv"

    def ms(m):
        v = m.get("mean")
        return "n/a" if v is None else f"{v}±{m.get('std', 0.0)}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["control_id", "level", "category", "subcategory", "severity",
                    "method", "orientation", "per_agent_detector",
                    "honest_subscore", "adversary_subscore", "adv_flag_rate",
                    "honest_fp_rate", "discriminative", "detects_attacks"])
        for r in sc["rows"]:
            w.writerow([r["id"], r["level"], r["category"], r["subcategory"],
                        r["severity"], r["method"], r["orient"],
                        r["per_agent_detector"], ms(r["honest_subscore"]),
                        ms(r["adversary_subscore"]), ms(r["adv_flag_rate"]),
                        ms(r["honest_fp_rate"]), r["discriminative"],
                        ";".join(r["detects"])])
        w.writerow([])
        w.writerow([f"discriminative={sc['n_discriminative']}",
                    f"attestation/system={sc['n_attestation_or_system']}",
                    f"total={len(sc['rows'])}"])
    return str(p)


def write_threshold_csv(ts: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_threshold_sweep.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["anomaly_z_threshold(tau)", "anomaly_recall", "anomaly_fpr",
                    "framework_recall", "framework_precision", "framework_fpr",
                    "framework_f1", "stealth_scale_detected"])
        for r in ts["rows"]:
            w.writerow([r["tau"], r["anomaly_recall"], r["anomaly_fpr"],
                        r["framework_recall"], r["framework_precision"],
                        r["framework_fpr"], r["framework_f1"],
                        r["stealth_detected"]])
        w.writerow([])
        w.writerow(["recommended_tau", ts["recommended_tau"]])
        w.writerow(["stealth_agents", ts["stealth_agents"]])
        w.writerow(["z_scores", ts["z_scores"]])
    return str(p)


def write_fraction_csv(fs: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_fraction_sweep.csv"

    def ms(m):
        return f"{m['mean']}±{m['std']}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["k_malicious", "malicious_fraction", "naive_accuracy",
                    "governance_accuracy", "oracle_accuracy", "detection_recall",
                    "detection_precision", "detection_fpr"])
        for r in fs["rows"]:
            w.writerow([r["k"], r["malicious_fraction"], ms(r["naive_accuracy"]),
                        ms(r["governance_accuracy"]), ms(r["oracle_accuracy"]),
                        ms(r["detection_recall"]), ms(r["detection_precision"]),
                        ms(r["detection_fpr"])])
    return str(p)


def write_control_ablation_csv(ca: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_control_ablation_full.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["control_id", "level", "category", "subcategory", "severity",
                    "method", "orientation", "load_bearing",
                    "n_newly_undetected", "newly_undetected_agents",
                    "accuracy_under_ablation", "delta_vs_governance"])
        for r in ca["rows"]:
            w.writerow([r["id"], r["level"], r["category"], r["subcategory"],
                        r["severity"], r["method"], r["orient"], r["load_bearing"],
                        r["n_newly_undetected"], ";".join(map(str, r["newly_undetected"])),
                        r["accuracy_under_ablation"], r["delta_vs_governance"]])
        w.writerow([])
        w.writerow([f"load_bearing={ca['n_load_bearing']}",
                    f"redundant(defense-in-depth)={ca['n_redundant']}",
                    f"governance_accuracy={ca['governance_accuracy']}"])
    return str(p)


def write_scalability_csv(sc: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_scalability.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["sweep", "num_agents", "num_rounds", "t_fl_train_s",
                    "t_governance_s", "t_chain_verify_s", "ledger_blocks",
                    "ledger_bytes", "detection_recall"])
        for r in sc["rows"]:
            w.writerow([r["sweep"], r["num_agents"], r["num_rounds"],
                        r["t_fl_train_s"], r["t_governance_s"],
                        r["t_chain_verify_s"], r["ledger_blocks"],
                        r["ledger_bytes"], r["detection_recall"]])
    return str(p)


def write_cross_dataset_csv(cd: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_cross_dataset.csv"

    def ms(m):
        return f"{m['mean']}±{m['std']}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "naive_accuracy", "governance_accuracy",
                    "oracle_accuracy", "gating_uplift", "detection_recall",
                    "detection_precision", "detection_f1"])
        for d in cd["datasets"]:
            e1, e2 = d["E1"], d["E2"]
            w.writerow([d["dataset"], ms(e1["naive_accuracy"]),
                        ms(e1["governance_accuracy"]), ms(e1["oracle_accuracy"]),
                        ms(e1["gating_uplift"]), ms(e2["recall"]),
                        ms(e2["precision"]), ms(e2["f1"])])
    return str(p)


def write_manifold_csv(mf: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_manifold.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        # detector comparison summary
        w.writerow(["detector", "recall_update_attacks", "false_positive_rate"])
        for name, d in mf["detectors"].items():
            w.writerow([name, d["recall_update_attacks"], d["fpr"]])
        w.writerow([])
        # per-agent
        w.writerow(["agent", "behaviour", "norm_z", "cosine_z",
                    "is_update_adversary", "caught_by"])
        for a in mf["per_agent"]:
            w.writerow([a["agent"], a["behaviour"], a["norm_z"], a["cosine_z"],
                        a["is_update_adversary"], "+".join(a["caught_by"]) or "-"])
        w.writerow([])
        # post-training distribution health
        w.writerow(["post_training_health", "pred_class_kl", "max_class_share",
                    "pred_class_entropy", "healthy"])
        for name in ("naive_model", "governance_gated_model"):
            h = mf["post_training_health"][name]
            w.writerow([name, h["pred_class_kl"], h["max_class_share"],
                        h["pred_class_entropy"], h["healthy"]])
    return str(p)


def write_noniid_csv(ni: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_noniid.csv"

    def ms(m):
        return f"{m['mean']}±{m['std']}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dirichlet_alpha", "heterogeneity", "naive_accuracy",
                    "governance_accuracy", "oracle_accuracy", "adversary_recall",
                    "honest_false_positive_rate"])
        for r in ni["rows"]:
            het = "IID-like" if r["alpha"] >= 10 else (
                "mild" if r["alpha"] >= 1 else ("strong" if r["alpha"] >= 0.5 else "extreme"))
            w.writerow([r["alpha"], het, ms(r["naive_accuracy"]),
                        ms(r["governance_accuracy"]), ms(r["oracle_accuracy"]),
                        ms(r["adversary_recall"]), ms(r["honest_false_positive_rate"])])
    return str(p)


def write_contribution_csv(c: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_contribution.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "behaviour", "is_adversary",
                    "marginal_contribution(mean±std)"])
        for r in c["rows"]:
            m = r["marginal_contribution"]
            w.writerow([r["agent"], r["behaviour"], r["is_adversary"],
                        f"{m['mean']}±{m['std']}"])
    return str(p)


def write_model_quality_csv(q: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_model_quality.csv"

    def ms(m):
        return f"{m['mean']}±{m['std']}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "macro_f1", "worst_class_recall", "ece(calibration)"])
        for name in ("naive", "gated"):
            d = q[name]
            w.writerow([name, ms(d["macro_f1"]), ms(d["worst_class_recall"]),
                        ms(d["ece"])])
    return str(p)


def write_hierarchy_csv(h: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_hierarchy.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        # (a) level/category roll-up
        w.writerow(["== score roll-up (sub->category->level) =="])
        w.writerow(["level", "level_weight", "level_score", "category", "category_score"])
        for lvl, d in h["rollup"].items():
            for c, cs in d["categories"].items():
                w.writerow([lvl, d["weight"], d["level_score"], c, cs])
        w.writerow([])
        # (b) per-agent x level
        w.writerow(["== per-agent x level subscore =="])
        w.writerow(["agent", "behaviour", "is_adversary", "LARGE", "MEDIUM",
                    "SMALL", "verdict"])
        for aid, a in sorted(h["per_agent_level"].items(), key=lambda x: int(x[0])):
            lv = a["levels"]
            w.writerow([aid, a["behaviour"], a["is_adversary"], lv["LARGE"],
                        lv["MEDIUM"], lv["SMALL"], a["verdict"]])
        w.writerow([])
        # (c) detection ownership
        w.writerow(["== per-level detection ownership =="])
        w.writerow(["adversary", "behaviour", "flagged_by_levels", "controls"])
        for aid, d in h["detection_ownership"].items():
            w.writerow([aid, d["behaviour"], "+".join(d["flagged_by_levels"]),
                        ";".join(d["controls"])])
        w.writerow([])
        # (d) level ablation
        w.writerow(["== level ablation (disable a whole tier) =="])
        w.writerow(["level", "n_controls", "newly_undetected_agents",
                    "undetected_behaviours", "accuracy_under_ablation",
                    "delta_vs_governance"])
        for r in h["level_ablation"]:
            w.writerow([r["level"], r["n_controls"],
                        ";".join(map(str, r["newly_undetected"])),
                        ";".join(r["undetected_behaviours"]),
                        r["accuracy_under_ablation"], r["delta_vs_governance"]])
    return str(p)


def write_noniid_fix_csv(nf: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_noniid_fix.csv"

    def ms(m):
        return f"{m['mean']}±{m['std']}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dirichlet_alpha",
                    "labelprior_honest_fpr", "labelprior_recall",
                    "fltrust_honest_fpr", "fltrust_recall"])
        for r in nf["rows"]:
            w.writerow([r["alpha"], ms(r["labelprior_honest_fpr"]),
                        ms(r["labelprior_recall"]), ms(r["fltrust_honest_fpr"]),
                        ms(r["fltrust_recall"])])
    return str(p)


def write_backbone_csv(bb: dict, out_dir: Path | None = None) -> str:
    out_dir = Path(out_dir or config.REPORTS_DIR)
    p = out_dir / "exp_backbone.csv"

    def ms(m):
        return f"{m['mean']}±{m['std']}"

    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "naive_accuracy", "governance_accuracy",
                    "oracle_accuracy", "detection_recall",
                    "honest_false_positive_rate"])
        for r in bb["rows"]:
            w.writerow([r["backbone"], ms(r["naive_accuracy"]),
                        ms(r["governance_accuracy"]), ms(r["oracle_accuracy"]),
                        ms(r["detection_recall"]), ms(r["honest_false_positive_rate"])])
    return str(p)


def render_experiment_summary(exp: dict) -> str:
    e1 = exp["summary"]["E1_closed_loop"]
    e2 = exp["summary"]["E2_detection"]
    lines = [
        "# Governance Experiments — Summary",
        f"- Seeds: {exp['seeds']}",
        "",
        "## E1 Closed-loop trust gating (global FL accuracy)",
        f"- naive (all agents):        **{e1['naive_accuracy']['mean']} ± {e1['naive_accuracy']['std']}**",
        f"- governance (quarantine flagged): **{e1['governance_accuracy']['mean']} ± {e1['governance_accuracy']['std']}**",
        f"- oracle (ground-truth):     **{e1['oracle_accuracy']['mean']} ± {e1['oracle_accuracy']['std']}**",
        f"- gating uplift:             **{e1['gating_uplift']['mean']} ± {e1['gating_uplift']['std']}**",
        "",
        "## E2 Detection quality (bad-agent identification)",
        f"- precision **{e2['precision']['mean']}±{e2['precision']['std']}** · "
        f"recall **{e2['recall']['mean']}±{e2['recall']['std']}** · "
        f"F1 **{e2['f1']['mean']}±{e2['f1']['std']}** · "
        f"FPR **{e2['false_positive_rate']['mean']}±{e2['false_positive_rate']['std']}**",
        "",
        "## E3 Control ablation — what breaks when a control is removed",
        "| Protection removed | Slide-4 risk | Acc (mean±std) | Δ vs gov | Undetected adversaries |",
        "|---|---|---:|---:|---:|",
    ]
    for key, v in exp["summary"]["E3_ablation"].items():
        acc, dl, und = (v["accuracy_under_ablation"], v["delta_vs_governance"],
                        v["n_undetected_adversaries"])
        lines.append(f"| {v['name']} | {v['risk']} | {acc['mean']}±{acc['std']} | "
                     f"{dl['mean']} | {und['mean']}±{und['std']} |")
    return "\n".join(lines)


def render_markdown_summary(results: dict) -> str:
    v = results["verdict"]
    lines = [
        "# Agent Trustworthiness Governance — Run Summary",
        "",
        f"- Dataset: **{v.get('dataset')}**",
        f"- Global FL accuracy — naive FedAvg (all agents): **{v.get('global_accuracy')}**",
        f"- Global FL accuracy — trust-gated (untrusted excluded): "
        f"**{v.get('gated_accuracy')}**  (uplift **{v.get('accuracy_uplift_from_gating')}**)",
        f"- System foundation score: **{v['foundation_score']}** "
        f"(gate {'OPEN' if v['gate_ok'] else 'BLOCKED ' + str(v['gate_reasons'])})",
        f"- Verdicts: {v['verdict_counts']}",
        "",
        "| Agent | Principal | Injected behaviour | Reputation | Verdict | Failed controls (severity) |",
        "|------:|-----------|--------------------|-----------:|---------|----------------------------|",
    ]
    for aid in sorted(v["agents"], key=lambda x: int(x)):
        a = v["agents"][aid]
        fc = ", ".join(f"{k}({s})" for k, s in a["failed_controls"].items())
        lines.append(
            f"| {a['agent_id']} | {a['principal']} | {a['behaviour']} | "
            f"{a['trust_score']} | {a['verdict']} | {fc or '—'} |")
    return "\n".join(lines)
