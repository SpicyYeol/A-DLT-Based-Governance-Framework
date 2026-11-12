"""Run the paper experiments (E1 closed-loop, E2 detection, E3 ablation, E4
multi-seed) without Airflow, and write the CSV tables.

    python scripts/run_experiments.py            # uses config.exp_seeds()
    TRUST_EXP_SEEDS=3 python scripts/run_experiments.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("TRUST_DATA_ROOT",
                      str(Path(__file__).resolve().parents[1] / "data"))


def main() -> int:
    from trust import experiments, config
    from trust import report as rpt
    import json

    seeds = config.exp_seeds()
    print(f"Running E1-E4 over seeds {seeds} ...")
    exp = experiments.run_all(seeds)
    paths = rpt.write_experiment_csvs(exp)
    (config.REPORTS_DIR / "experiments.json").write_text(
        json.dumps(exp, indent=2, default=str), encoding="utf-8")

    print("Running C3 baseline aggregation comparison ...")
    bl = experiments.run_baselines(seeds)
    paths["baselines"] = rpt.write_baseline_csv(bl)

    print("Running C6 stealth / collusion profile ...")
    st = experiments.run_stealth(seeds)
    paths["stealth"] = rpt.write_stealth_csv(st)

    print("Running C5 threshold sensitivity (ROC) ...")
    ts = experiments.run_threshold_sweep(seeds[0])
    paths["threshold"] = rpt.write_threshold_csv(ts)

    print("Running C2 adversary-fraction sweep (5 seeds) ...")
    fs = experiments.run_fraction_sweep(seeds=seeds)
    paths["fraction"] = rpt.write_fraction_csv(fs)

    print("Running C4 full 36-control ablation ...")
    ca = experiments.run_control_ablation_full(seeds[0])
    paths["control_ablation"] = rpt.write_control_ablation_csv(ca)

    print("Running C7 scalability / overhead ...")
    sc = experiments.run_scalability(seed=seeds[0])
    paths["scalability"] = rpt.write_scalability_csv(sc)

    print("Running C8 cross-dataset generalization ...")
    cd = experiments.run_cross_dataset(seeds=seeds)
    paths["cross_dataset"] = rpt.write_cross_dataset_csv(cd)

    print("Running C9 manifold / post-training distribution checks ...")
    mf = experiments.run_manifold(seeds[0])
    paths["manifold"] = rpt.write_manifold_csv(mf)

    print("Running B3 non-IID severity sweep (5 seeds) ...")
    ni = experiments.run_noniid(seeds=seeds)
    paths["noniid"] = rpt.write_noniid_csv(ni)

    print("Running deep-model backbone (linear vs MLP, 5 seeds) ...")
    bb = experiments.run_deep_backbone(seeds=seeds)
    paths["backbone"] = rpt.write_backbone_csv(bb)

    print("Running B3-fix: non-IID-aware FLTrust drift control (5 seeds) ...")
    nf = experiments.run_noniid_fix(seeds=seeds)
    paths["noniid_fix"] = rpt.write_noniid_fix_csv(nf)

    print("Running H1 hierarchical (3-tier) experiment ...")
    hi = experiments.run_hierarchy(seeds[0])
    paths["hierarchy"] = rpt.write_hierarchy_csv(hi)

    print("Running sub-category capability check (36 controls, 5 seeds) ...")
    subc = experiments.run_subcategory_capability(seeds)
    paths["subcategory"] = rpt.write_subcategory_capability_csv(subc)

    from trust import ml_eval
    print("Running DL/ML contribution valuation (leave-one-out) ...")
    con = ml_eval.run_contribution(seeds=seeds[:3])
    paths["contribution"] = rpt.write_contribution_csv(con)

    print("Running DL/ML model-quality suite (macro-F1 / worst-class / ECE) ...")
    mq = ml_eval.run_model_quality(seeds=seeds)
    paths["model_quality"] = rpt.write_model_quality_csv(mq)

    print()
    print(rpt.render_experiment_summary(exp))
    print()
    print("## C3 Baseline aggregation (global accuracy, mean±std)")
    for name, m in bl["summary_accuracy"].items():
        print(f"  {name:20s} {m['mean']}±{m['std']}")
    print()
    print("## C6 Stealth/collusion detection rate by behaviour")
    print(f"  closed-loop acc: naive {st['E1_closed_loop']['naive_accuracy']['mean']} "
          f"-> gov {st['E1_closed_loop']['governance_accuracy']['mean']} | "
          f"recall {st['E2_detection']['recall']['mean']}")
    for beh, rate in st["per_behaviour_detection_rate"].items():
        print(f"  {beh:16s} detected {rate}")
    print()
    print("## C5 Threshold sweep (anomaly z tau) — stealth detection vs FPR")
    for r in ts["rows"]:
        print(f"  tau={r['tau']:<4} anomaly_recall={r['anomaly_recall']} "
              f"fw_recall={r['framework_recall']} fw_fpr={r['framework_fpr']} "
              f"stealth_caught={r['stealth_detected']}")
    print(f"  -> recommended tau = {ts['recommended_tau']}")
    print()
    print("## C2 Adversary-fraction sweep")
    print("  k  frac  naive   gov     recall")
    for r in fs["rows"]:
        print(f"  {r['k']}  {r['malicious_fraction']:<5} "
              f"{r['naive_accuracy']['mean']:<7} {r['governance_accuracy']['mean']:<7} "
              f"{r['detection_recall']['mean']}")
    print()
    print(f"## C4 Full control ablation: {ca['n_load_bearing']} load-bearing / "
          f"{ca['n_redundant']} redundant (defense-in-depth)")
    for r in ca["rows"]:
        if r["load_bearing"]:
            print(f"  {r['id']:8s} {r['subcategory']:22s} -> escapes {r['newly_undetected']} "
                  f"(acc {r['accuracy_under_ablation']}, Δ{r['delta_vs_governance']})")
    print()
    print("## C7 Scalability (t_fl / t_gov / ledger blocks)")
    for r in sc["rows"]:
        print(f"  {r['sweep']:6s} agents={r['num_agents']:<3} rounds={r['num_rounds']:<2} "
              f"fl={r['t_fl_train_s']}s gov={r['t_governance_s']}s "
              f"blocks={r['ledger_blocks']} recall={r['detection_recall']}")
    print()
    print("## C8 Cross-dataset generalization")
    for d in cd["datasets"]:
        print(f"  {d['dataset']:28s} naive {d['E1']['naive_accuracy']['mean']} -> "
              f"gov {d['E1']['governance_accuracy']['mean']} | recall {d['E2']['recall']['mean']}")
    print()
    print("## C9 Manifold / post-training distribution checks")
    for name, d in mf["detectors"].items():
        print(f"  detector {name:14s} recall={d['recall_update_attacks']} fpr={d['fpr']}")
    hn = mf["post_training_health"]["naive_model"]
    hg = mf["post_training_health"]["governance_gated_model"]
    print(f"  post-training health: naive KL={hn['pred_class_kl']} healthy={hn['healthy']} | "
          f"gated KL={hg['pred_class_kl']} healthy={hg['healthy']}")
    print()
    print("## B3 Non-IID severity (Dirichlet alpha)")
    print("  alpha  naive   gov     adv_recall  honest_FPR")
    for r in ni["rows"]:
        print(f"  {r['alpha']:<5} {r['naive_accuracy']['mean']:<7} "
              f"{r['governance_accuracy']['mean']:<7} "
              f"{r['adversary_recall']['mean']:<11} {r['honest_false_positive_rate']['mean']}")
    print()
    print("## Deep-model backbone")
    for r in bb["rows"]:
        print(f"  {r['backbone']:7s} naive {r['naive_accuracy']['mean']} -> "
              f"gov {r['governance_accuracy']['mean']} | recall {r['detection_recall']['mean']} "
              f"| honest_FPR {r['honest_false_positive_rate']['mean']}")
    print()
    print("## B3-fix: label-prior drift vs FLTrust (honest FPR / directional recall)")
    for r in nf["rows"]:
        print(f"  alpha={r['alpha']:<5} label-prior FPR={r['labelprior_honest_fpr']['mean']} "
              f"rec={r['labelprior_recall']['mean']}  |  FLTrust FPR={r['fltrust_honest_fpr']['mean']} "
              f"rec={r['fltrust_recall']['mean']}")
    print()
    print("## H1 Hierarchy — level scores + which tier flags each attack")
    for lvl, d in hi["rollup"].items():
        print(f"  {lvl:6s} score={d['level_score']} (weight {d['weight']})")
    for aid, d in hi["detection_ownership"].items():
        print(f"  agent {aid} {d['behaviour']:14s} flagged by {'+'.join(d['flagged_by_levels']) or '-'}")
    print("  level ablation:")
    for r in hi["level_ablation"]:
        print(f"    disable {r['level']:6s} -> escapes {r['undetected_behaviours']} "
              f"(acc {r['accuracy_under_ablation']}, Δ{r['delta_vs_governance']})")
    print()
    print(f"## Sub-category capability: {subc['n_discriminative']}/36 discriminative, "
          f"{subc['n_attestation_or_system']} attestation/system")
    for r in subc["rows"]:
        if r["discriminative"]:
            print(f"  {r['id']:8s} {r['subcategory']:22s} adv={r['adversary_subscore']['mean']} "
                  f"honest={r['honest_subscore']['mean']} detects={r['detects']}")
    print()
    print("## DL/ML contribution valuation (leave-one-out marginal accuracy)")
    for r in con["rows"]:
        print(f"  ag{r['agent']} {r['behaviour']:12s} marginal={r['marginal_contribution']['mean']:+.3f}")
    print("## DL/ML model quality (naive vs gated)")
    print(f"  naive: macroF1={mq['naive']['macro_f1']['mean']} worstClass={mq['naive']['worst_class_recall']['mean']} ECE={mq['naive']['ece']['mean']}")
    print(f"  gated: macroF1={mq['gated']['macro_f1']['mean']} worstClass={mq['gated']['worst_class_recall']['mean']} ECE={mq['gated']['ece']['mean']}")
    print()
    for k, v in paths.items():
        print(f"{k:12s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
