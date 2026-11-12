# -*- coding: utf-8 -*-
"""Human-readable summaries rendered as Markdown.

Split out of the single report module; the code is unchanged.
"""
from __future__ import annotations

from ..taxonomy import ALL_CONTROLS


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
        "| Protection removed | Risk | Acc (mean±std) | Δ vs gov | Undetected adversaries |",
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
