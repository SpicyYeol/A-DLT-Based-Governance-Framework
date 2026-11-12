"""Paper experiments for the Agent-Trustworthiness Governance framework.

Implements four experiment families and writes paper-ready CSVs:

  E1  Closed-loop trust gating (RQ: does the framework's *own* verdict, used to
      gate federated aggregation, recover model accuracy?). Compares
        naive       – aggregate all agents (no governance)
        governance  – quarantine every agent the framework flags (Under Review
                      or Untrusted) from aggregation  [CLOSED LOOP]
        oracle      – exclude the ground-truth-bad agents (upper bound)

  E2  Detection quality (RQ: how well are bad agents identified?). Precision /
      recall / F1 / false-positive-rate of "flagged" vs ground truth, with a
      confusion matrix.

  E3  Control ablation (RQ: what problem appears when a control is removed?).
      For each protection group we disable it and measure the harm it was
      preventing — mapped to the deck's slide-4 "Risks of not verifying
      untrustworthy agents". Two harm axes are reported:
        * accuracy_under_ablation  – model accuracy once the now-undetected
          agents are re-admitted to aggregation (matters for poisoning);
        * n_undetected_adversaries – bad agents that flip to "Trusted" and thus
          escape governance entirely (matters for integrity/freshness/identity).

  E4  Multi-seed variance (RQ: are the results stable?). E1–E3 are repeated over
      several seeds and reported as mean +/- std.

Governance gating policy
------------------------
Aggregation excludes any *flagged* agent (verdict != "Trusted"): Untrusted are
rejected/suspended, Under Review are quarantined pending audit. This conservative
policy is what makes the closed-loop accuracy comparison and the ablations
meaningful.
"""

from __future__ import annotations

import statistics as stats

from . import config
from . import fl_simulation as fl
from . import aggregators as agg
from .engine import build_ctx, evaluate_controls, agent_verdict
from .ledger import Ledger


# Protection groups to ablate. Each disables one or more controls and states the
# slide-4 risk it guards against and which harm axis is primary.
ABLATIONS = [
    {"key": "no_governance", "disabled": set(), "aggregate_all": True,
     "name": "No trust governance (aggregate all agents)",
     "risk": "Degraded model accuracy + security", "primary": "accuracy"},
    {"key": "no_model_poison_defense", "disabled": {"M-BC-01", "M-SC-02"},
     "name": "No model-poisoning detection (M-BC-01 anomaly, M-SC-02 over-weight)",
     "risk": "Degraded model accuracy (gradient/scale poisoning)",
     "primary": "accuracy"},
    {"key": "no_data_poison_defense", "disabled": {"S-DF-03"},
     "name": "No data-quality drift check (S-DF-03, ISO/IEC 5259)",
     "risk": "Degraded accuracy (label/data poisoning)", "primary": "accuracy"},
    {"key": "no_freshness", "disabled": {"S-DF-02"},
     "name": "No data-freshness SLA (S-DF-02)",
     "risk": "Stale / replayed data accepted", "primary": "undetected"},
    {"key": "no_message_integrity", "disabled": {"S-MI-02", "M-CV-03"},
     "name": "No payload-hash + signature verification (S-MI-02, M-CV-03)",
     "risk": "Tampered / forged submissions undetected", "primary": "undetected"},
    {"key": "no_identity_lifecycle", "disabled": {"L-IA-03", "M-CV-02"},
     "name": "No identity-lifecycle / revocation (L-IA-03, M-CV-02)",
     "risk": "Revoked agents keep contributing", "primary": "undetected"},
]


def _flagged_set(results, agent_ids, disabled=frozenset()) -> set[int]:
    """Agents the framework would quarantine (verdict != Trusted)."""
    return {aid for aid in agent_ids
            if agent_verdict(results, aid, disabled=disabled)[0] != "Trusted"}


def _untrusted_only(results, agent_ids, disabled=frozenset()) -> set[int]:
    return {aid for aid in agent_ids
            if agent_verdict(results, aid, disabled=disabled)[0] == "Untrusted"}


def _conf(flagged: set[int], agent_ids, ground_truth_bad: set[int]) -> dict:
    good = set(agent_ids) - ground_truth_bad
    tp = len(flagged & ground_truth_bad)
    fp = len(flagged & good)
    fn = len(ground_truth_bad - flagged)
    tn = len(good - flagged)
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "false_positive_rate": round(fpr, 4)}


def detection_metrics(results, agent_ids, ground_truth_bad: set[int]) -> dict:
    return _conf(_flagged_set(results, agent_ids), agent_ids, ground_truth_bad)


def run_seed(seed: int, profile: dict | None = None,
             dataset: str | None = None, ablations: bool = True) -> dict:
    """Run E1–E3 for a single seed. Returns a JSON-able result dict.

    ``profile`` overrides the adversary profile (C6); ``dataset`` selects the
    image dataset (C8); ``ablations=False`` skips the E3 retrains (C8)."""
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed, dataset=dataset)

    # reference (naive) pass: trains all-agent model + records ledger & signals
    naive = fl.train_trajectory(prep, exclude=frozenset(), record=True,
                                reset_ledger=True, profile=profile)
    signals = naive["signals"]

    ctx = build_ctx(signals)               # reads the ledger just written
    results = evaluate_controls(ctx)       # cid -> ControlResult
    ids = ctx.agent_ids()
    gt_bad = set(profile) | {config.REVOKED_AGENT}

    flagged = _flagged_set(results, ids)
    # ---- E1 closed-loop gating ----
    naive_acc = naive["accuracy"]
    gov_acc = fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"]
    oracle_acc = fl.train_trajectory(prep, exclude=gt_bad, profile=profile)["accuracy"]

    # ---- E2 detection metrics ----
    det = detection_metrics(results, ids, gt_bad)

    # per-agent verdict snapshot (for reference tables)
    verdicts = {}
    for aid in ids:
        v, failed = agent_verdict(results, aid)
        verdicts[aid] = {"behaviour": ctx.agents[aid].get("behaviour"),
                         "is_adversary": aid in gt_bad, "verdict": v,
                         "failed_controls": failed}

    # ---- E3 ablations ----
    ablation_rows = []
    for ab in (ABLATIONS if ablations else []):
        disabled = frozenset(ab["disabled"])
        if ab.get("aggregate_all"):
            abl_flagged = set()               # nothing quarantined
        else:
            abl_flagged = _flagged_set(results, ids, disabled=disabled)
        acc = naive_acc if ab.get("aggregate_all") else \
            fl.train_trajectory(prep, exclude=abl_flagged, profile=profile)["accuracy"]
        # bad agents that escape detection entirely under this ablation
        undetected = sorted(gt_bad - _flagged_set(results, ids, disabled=disabled)) \
            if not ab.get("aggregate_all") else sorted(gt_bad)
        ablation_rows.append({
            "key": ab["key"], "name": ab["name"], "risk": ab["risk"],
            "primary_harm": ab["primary"],
            "accuracy_under_ablation": round(acc, 4),
            "delta_vs_governance": round(acc - gov_acc, 4),
            "n_undetected_adversaries": len(undetected),
            "undetected_agents": undetected,
        })

    return {
        "seed": seed, "dataset": signals["dataset"],
        "naive_accuracy": round(naive_acc, 4),
        "governance_accuracy": round(gov_acc, 4),
        "oracle_accuracy": round(oracle_acc, 4),
        "gating_uplift": round(gov_acc - naive_acc, 4),
        "flagged_agents": sorted(flagged),
        "detection": det,
        "verdicts": verdicts,
        "ablations": ablation_rows,
    }


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    return {"mean": round(stats.mean(values), 4),
            "std": round(stats.pstdev(values), 4) if len(values) > 1 else 0.0,
            "n": len(values)}


def run_all(seeds: list[int] | None = None) -> dict:
    seeds = seeds or config.exp_seeds()
    per_seed = [run_seed(s) for s in seeds]

    # aggregate E1
    e1 = {k: _agg([r[k] for r in per_seed]) for k in
          ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
           "gating_uplift")}
    # aggregate E2
    e2 = {k: _agg([r["detection"][k] for r in per_seed]) for k in
          ("precision", "recall", "f1", "false_positive_rate")}
    # aggregate E3 by ablation key
    e3 = {}
    for ab in ABLATIONS:
        rows = [next(x for x in r["ablations"] if x["key"] == ab["key"])
                for r in per_seed]
        e3[ab["key"]] = {
            "name": ab["name"], "risk": ab["risk"],
            "primary_harm": ab["primary"],
            "accuracy_under_ablation": _agg([x["accuracy_under_ablation"] for x in rows]),
            "delta_vs_governance": _agg([x["delta_vs_governance"] for x in rows]),
            "n_undetected_adversaries": _agg([x["n_undetected_adversaries"] for x in rows]),
        }
    return {"seeds": seeds, "per_seed": per_seed,
            "summary": {"E1_closed_loop": e1, "E2_detection": e2,
                        "E3_ablation": e3}}


# ---------------------------------------------------------------------------
# C3 — baseline robust-aggregation comparison
# ---------------------------------------------------------------------------
def run_baselines_seed(seed: int, profile: dict | None = None) -> dict:
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | {config.REVOKED_AGENT}
    flagged = _flagged_set(results, ids)

    acc = {}
    for name, fn in agg.BASELINES.items():          # robust aggregators, all agents
        acc[name] = round(fl.train_trajectory(
            prep, profile=profile, aggregator=fn)["accuracy"], 4)
    acc["Governance (ours)"] = round(fl.train_trajectory(
        prep, exclude=flagged, profile=profile)["accuracy"], 4)
    acc["Oracle"] = round(fl.train_trajectory(
        prep, exclude=gt_bad, profile=profile)["accuracy"], 4)

    return {"seed": seed, "accuracy": acc,
            "detection_ours": detection_metrics(results, ids, gt_bad),
            "flagged": sorted(flagged), "gt_bad": sorted(gt_bad)}


def run_baselines(seeds: list[int] | None = None, profile: dict | None = None) -> dict:
    seeds = seeds or config.exp_seeds()
    per_seed = [run_baselines_seed(s, profile) for s in seeds]
    names = list(per_seed[0]["accuracy"].keys())
    summary = {n: _agg([r["accuracy"][n] for r in per_seed]) for n in names}
    det = {k: _agg([r["detection_ours"][k] for r in per_seed]) for k in
           ("precision", "recall", "f1", "false_positive_rate")}
    return {"seeds": seeds, "per_seed": per_seed,
            "summary_accuracy": summary, "detection_ours": det}


# ---------------------------------------------------------------------------
# C6 — stealth / collusion attack profile
# ---------------------------------------------------------------------------
def run_stealth(seeds: list[int] | None = None) -> dict:
    """Run E1/E2 under the harder STEALTH_PROFILE (low-magnitude scale +
    colluding label-flip) to probe detection blind spots."""
    seeds = seeds or config.exp_seeds()
    profile = config.STEALTH_PROFILE
    per_seed = [run_seed(s, profile=profile) for s in seeds]
    e1 = {k: _agg([r[k] for r in per_seed]) for k in
          ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
           "gating_uplift")}
    e2 = {k: _agg([r["detection"][k] for r in per_seed]) for k in
          ("precision", "recall", "f1", "false_positive_rate")}
    # per-behaviour detection: was each adversary flagged?
    behaviour_detection = {}
    for r in per_seed:
        for aid, v in r["verdicts"].items():
            if v["is_adversary"]:
                beh = v["behaviour"]
                behaviour_detection.setdefault(beh, []).append(
                    1 if v["verdict"] != "Trusted" else 0)
    beh_rate = {b: round(sum(x) / len(x), 3) for b, x in behaviour_detection.items()}
    beh_std = {b: round(_agg(x)["std"], 3) for b, x in behaviour_detection.items()}
    return {"seeds": seeds, "profile": {str(k): v for k, v in profile.items()},
            "per_seed": per_seed, "E1_closed_loop": e1, "E2_detection": e2,
            "per_behaviour_detection_rate": beh_rate,
            "per_behaviour_detection_std": beh_std}


# ---------------------------------------------------------------------------
# C5 — threshold sensitivity (ROC) for the update-anomaly control
# ---------------------------------------------------------------------------
def _anomaly_z(signals: dict) -> dict:
    """Robust modified z-score of each agent's mean update norm (same statistic
    M-BC-01 uses), computed straight from recorded signals."""
    import numpy as np
    agents = {int(k): v for k, v in signals["agents"].items()}
    ids = sorted(agents)
    norms = np.array([np.mean([r["update_norm"] for r in agents[i]["rounds"]])
                      for i in ids])
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) or 1e-9
    z = 0.6745 * (norms - med) / mad
    return {i: float(abs(zi)) for i, zi in zip(ids, z)}


def run_threshold_sweep(seed: int, taus: list[float] | None = None,
                        profile: dict | None = None) -> dict:
    """Sweep the anomaly z-threshold tau under the stealth profile and report
    detection ROC — both for the anomaly control alone and for the full
    framework (anomaly retuned + all other controls)."""
    profile = config.STEALTH_PROFILE if profile is None else profile
    taus = taus or [4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5]
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    zmap = _anomaly_z(signals)
    stealth_ids = {a for a, b in profile.items() if b == "stealth_scale"}

    rows = []
    for tau in taus:
        # anomaly detector alone: flag if |z| > tau
        anomaly_flagged = {i for i in ids if zmap[i] > tau}
        ano = _conf(anomaly_flagged, ids, gt_bad)
        # full framework: override M-BC-01 per-agent at this tau, re-verdict
        base = {i: results["M-BC-01"].per_agent.get(i, 1.0) for i in ids}
        for i in ids:
            results["M-BC-01"].per_agent[i] = 0.0 if zmap[i] > tau else 1.0
        fw_flagged = _flagged_set(results, ids)
        fw = _conf(fw_flagged, ids, gt_bad)
        for i in ids:                       # restore
            results["M-BC-01"].per_agent[i] = base[i]
        rows.append({
            "tau": tau,
            "anomaly_recall": ano["recall"], "anomaly_fpr": ano["false_positive_rate"],
            "framework_recall": fw["recall"], "framework_precision": fw["precision"],
            "framework_fpr": fw["false_positive_rate"], "framework_f1": fw["f1"],
            "stealth_detected": bool(stealth_ids <= anomaly_flagged),
        })
    # recommend the most conservative (largest) tau that maximises framework F1
    # while keeping FPR == 0 -> closes the stealth gap with fewest interventions
    zero_fp = [r for r in rows if r["framework_fpr"] == 0.0]
    pick = max(zero_fp or rows, key=lambda r: (r["framework_f1"], r["tau"]))
    return {"seed": seed, "z_scores": {str(k): round(v, 2) for k, v in zmap.items()},
            "stealth_agents": sorted(stealth_ids), "rows": rows,
            "recommended_tau": pick["tau"], "recommended": pick}


# ---------------------------------------------------------------------------
# C2 — adversary-fraction sweep
# ---------------------------------------------------------------------------
_ATTACK_ORDER = [1, 3, 5, 6, 2, 4, 0]          # slots to fill as k grows
_ATTACK_CYCLE = ["scale_attack", "label_flip", "stale", "tamper"]


def make_profile(k: int) -> dict:
    return {_ATTACK_ORDER[i]: _ATTACK_CYCLE[i % len(_ATTACK_CYCLE)]
            for i in range(min(k, len(_ATTACK_ORDER)))}


def run_fraction_sweep(ks: list[int] | None = None,
                       seeds: list[int] | None = None) -> dict:
    """Vary the number of malicious agents k and measure accuracy + detection.
    Disables the standalone revoked-agent during the sweep so ground truth is
    exactly the k injected attackers."""
    n = config.NUM_AGENTS
    ks = ks or list(range(1, n))            # 1 .. n-1 malicious
    seeds = seeds or config.exp_seeds()[:2]  # keep the sweep bounded
    saved_revoked = config.REVOKED_AGENT
    config.REVOKED_AGENT = -1               # no standalone revocation here
    try:
        rows = []
        for k in ks:
            profile = make_profile(k)
            gt_bad = set(profile)
            per = {"naive": [], "gov": [], "oracle": [],
                   "recall": [], "precision": [], "fpr": []}
            for s in seeds:
                prep = fl.prepare(s)
                naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                            profile=profile)
                signals = naive["signals"]
                ctx = build_ctx(signals)
                results = evaluate_controls(ctx)
                ids = ctx.agent_ids()
                flagged = _flagged_set(results, ids)
                det = _conf(flagged, ids, gt_bad)
                per["naive"].append(naive["accuracy"])
                per["gov"].append(fl.train_trajectory(
                    prep, exclude=flagged, profile=profile)["accuracy"])
                per["oracle"].append(fl.train_trajectory(
                    prep, exclude=gt_bad, profile=profile)["accuracy"])
                per["recall"].append(det["recall"])
                per["precision"].append(det["precision"])
                per["fpr"].append(det["false_positive_rate"])
            rows.append({
                "k": k, "malicious_fraction": round(k / n, 3),
                "naive_accuracy": _agg(per["naive"]),
                "governance_accuracy": _agg(per["gov"]),
                "oracle_accuracy": _agg(per["oracle"]),
                "detection_recall": _agg(per["recall"]),
                "detection_precision": _agg(per["precision"]),
                "detection_fpr": _agg(per["fpr"]),
            })
    finally:
        config.REVOKED_AGENT = saved_revoked
    return {"seeds": seeds, "num_agents": n, "rows": rows}


# ---------------------------------------------------------------------------
# C4 — full 36-control ablation (marginal value of every control)
# ---------------------------------------------------------------------------
def run_control_ablation_full(seed: int | None = None,
                              profile: dict | None = None) -> dict:
    from .taxonomy import ALL_CONTROLS
    seed = config.EXP_BASE_SEED if seed is None else seed
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    flagged_full = _flagged_set(results, ids)
    gov_acc = fl.train_trajectory(prep, exclude=flagged_full, profile=profile)["accuracy"]

    rows = []
    for cp in ALL_CONTROLS:
        flagged_ab = _flagged_set(results, ids, disabled=frozenset({cp.id}))
        # adversaries detected with the full control set that now escape
        newly = sorted((gt_bad & flagged_full) - flagged_ab)
        load_bearing = len(newly) > 0
        acc = (fl.train_trajectory(prep, exclude=flagged_ab, profile=profile)["accuracy"]
               if load_bearing else gov_acc)
        rows.append({
            "id": cp.id, "level": cp.level, "category": cp.category,
            "subcategory": cp.subcategory, "severity": cp.severity,
            "method": cp.method, "orient": cp.orient,
            "load_bearing": load_bearing,
            "n_newly_undetected": len(newly), "newly_undetected": newly,
            "accuracy_under_ablation": round(acc, 4),
            "delta_vs_governance": round(acc - gov_acc, 4),
        })
    n_load = sum(1 for r in rows if r["load_bearing"])
    return {"seed": seed, "governance_accuracy": round(gov_acc, 4),
            "flagged_full": sorted(flagged_full), "gt_bad": sorted(gt_bad),
            "n_load_bearing": n_load, "n_redundant": len(rows) - n_load,
            "rows": rows}


# ---------------------------------------------------------------------------
# C7 — scalability / overhead
# ---------------------------------------------------------------------------
def _measure_scale(seed: int) -> dict:
    import time
    from .ledger import Ledger
    profile = config.ADVERSARY_PROFILE
    t0 = time.perf_counter()
    prep = fl.prepare(seed)
    t_prep = time.perf_counter() - t0

    t0 = time.perf_counter()
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    t_fl = time.perf_counter() - t0
    signals = naive["signals"]

    t0 = time.perf_counter()
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    flagged = _flagged_set(results, ids)
    t_gov = time.perf_counter() - t0

    ledger = Ledger()
    t0 = time.perf_counter()
    chain = ledger.verify_chain()
    t_verify = time.perf_counter() - t0
    ledger_bytes = ledger.path.stat().st_size if ledger.path.exists() else 0

    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    det = _conf(flagged, ids, gt_bad)
    return {"t_prepare_s": round(t_prep, 3), "t_fl_train_s": round(t_fl, 3),
            "t_governance_s": round(t_gov, 4), "t_chain_verify_s": round(t_verify, 4),
            "ledger_blocks": chain["blocks"], "ledger_bytes": ledger_bytes,
            "detection_recall": det["recall"]}


def run_scalability(agent_counts: list[int] | None = None,
                    round_counts: list[int] | None = None,
                    seed: int | None = None) -> dict:
    seed = config.EXP_BASE_SEED if seed is None else seed
    agent_counts = agent_counts or [4, 8, 16, 32]
    round_counts = round_counts or [1, 2, 4, 8]
    saved_a, saved_r = config.NUM_AGENTS, config.NUM_ROUNDS
    rows = []
    try:
        for na in agent_counts:                 # sweep #agents (rounds fixed)
            config.NUM_AGENTS, config.NUM_ROUNDS = na, saved_r
            m = _measure_scale(seed)
            rows.append({"sweep": "agents", "num_agents": na,
                         "num_rounds": saved_r, **m})
        for nr in round_counts:                 # sweep #rounds (agents fixed)
            config.NUM_AGENTS, config.NUM_ROUNDS = saved_a, nr
            m = _measure_scale(seed)
            rows.append({"sweep": "rounds", "num_agents": saved_a,
                         "num_rounds": nr, **m})
    finally:
        config.NUM_AGENTS, config.NUM_ROUNDS = saved_a, saved_r
    return {"seed": seed, "rows": rows}


# ---------------------------------------------------------------------------
# C8 — cross-dataset generalization
# ---------------------------------------------------------------------------
def run_cross_dataset(datasets: list[str] | None = None,
                      seeds: list[int] | None = None) -> dict:
    datasets = datasets or ["mnist", "fashion_mnist"]
    seeds = seeds or config.exp_seeds()
    out = []
    for name in datasets:
        per = [run_seed(s, dataset=name, ablations=False) for s in seeds]
        e1 = {k: _agg([r[k] for r in per]) for k in
              ("naive_accuracy", "governance_accuracy", "oracle_accuracy",
               "gating_uplift")}
        e2 = {k: _agg([r["detection"][k] for r in per]) for k in
              ("precision", "recall", "f1", "false_positive_rate")}
        out.append({"dataset": per[0]["dataset"], "E1": e1, "E2": e2})
    return {"seeds": seeds, "datasets": out}


def run_subcategory_capability(seeds: list[int] | None = None,
                               profile: dict | None = None) -> dict:
    """Per-sub-category capability check (all 36). For each control, across
    seeds: mean subscore on honest vs on adversaries, adversary-flag rate,
    honest false-positive rate, and which attacks it actually catches.
    Reveals which sub-categories are active per-agent detectors vs
    system/attestation controls."""
    from .taxonomy import ALL_CONTROLS
    import statistics as st
    seeds = seeds or config.exp_seeds()
    profile = profile or config.ADVERSARY_PROFILE
    acc = {cp.id: {"honest": [], "adv": [], "fa": [], "fh": []} for cp in ALL_CONTROLS}
    detects = {cp.id: set() for cp in ALL_CONTROLS}
    for s in seeds:
        prep = fl.prepare(s)
        naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
        ctx = build_ctx(naive["signals"])
        results = evaluate_controls(ctx)
        ids = ctx.agent_ids()
        gt = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
        honest = set(ids) - gt
        for cp in ALL_CONTROLS:
            pa = results[cp.id].per_agent
            if not pa:
                continue
            hs = [pa[a] for a in honest if a in pa]
            ad = [pa[a] for a in gt if a in pa]
            if hs:
                acc[cp.id]["honest"].append(st.mean(hs))
            if ad:
                acc[cp.id]["adv"].append(st.mean(ad))
            acc[cp.id]["fa"].append(1.0 if any(pa.get(a, 1) < 0.5 for a in gt) else 0.0)
            acc[cp.id]["fh"].append(1.0 if any(pa.get(a, 1) < 0.5 for a in honest) else 0.0)
            for a in gt:
                if pa.get(a, 1) < 0.5:
                    lbl = ("revoked" if a == config.REVOKED_AGENT
                           else ctx.agents[a].get("behaviour", str(a)))
                    detects[cp.id].add(lbl)
    rows = []
    for cp in ALL_CONTROLS:
        a = acc[cp.id]
        per_agent_control = bool(a["honest"] or a["adv"])
        rows.append({
            "id": cp.id, "level": cp.level, "category": cp.category,
            "subcategory": cp.subcategory, "severity": cp.severity,
            "method": cp.method, "orient": cp.orient,
            "per_agent_detector": per_agent_control,
            "honest_subscore": _agg(a["honest"]) if a["honest"] else {"mean": 1.0, "std": 0.0},
            "adversary_subscore": _agg(a["adv"]) if a["adv"] else {"mean": None, "std": None},
            "adv_flag_rate": _agg(a["fa"]) if a["fa"] else {"mean": 0.0, "std": 0.0},
            "honest_fp_rate": _agg(a["fh"]) if a["fh"] else {"mean": 0.0, "std": 0.0},
            "detects": sorted(detects[cp.id]),
            "discriminative": bool(detects[cp.id])})
    n_disc = sum(1 for r in rows if r["discriminative"])
    return {"seeds": seeds, "n_discriminative": n_disc,
            "n_attestation_or_system": len(rows) - sum(1 for r in rows
                                                       if r["per_agent_detector"]),
            "rows": rows}


def run_hierarchy(seed: int | None = None, profile: dict | None = None) -> dict:
    """H1 — demonstrate the 3-tier hierarchy (Clause 6). Produces:
      (a) score roll-up  sub-category -> category -> level;
      (b) per-agent × level detection contribution (which tier flags whom);
      (c) per-level detection ownership (which tier catches which attack);
      (d) level ablation — disable a whole tier, show it owns distinct threats.
    """
    from .taxonomy import by_level, by_id, LEVEL_WEIGHTS
    import statistics as st
    seed = config.EXP_BASE_SEED if seed is None else seed
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    flagged_full = _flagged_set(results, ids)
    gov_acc = fl.train_trajectory(prep, exclude=flagged_full, profile=profile)["accuracy"]
    levels = ("LARGE", "MEDIUM", "SMALL")

    # (a) score roll-up
    rollup = {}
    for lvl in levels:
        cats = {}
        for cp in by_level(lvl):
            cats.setdefault(cp.category, []).append(results[cp.id].score)
        cat_means = {c: round(st.mean(v), 3) for c, v in cats.items()}
        rollup[lvl] = {"categories": cat_means,
                       "level_score": round(st.mean(list(cat_means.values())), 3),
                       "weight": LEVEL_WEIGHTS[lvl]}

    # (b) per-agent × level subscore + verdict
    per_agent_level = {}
    for aid in ids:
        d = {}
        for lvl in levels:
            subs = [results[cp.id].per_agent[aid] for cp in by_level(lvl)
                    if aid in results[cp.id].per_agent]
            d[lvl] = round(st.mean(subs), 3) if subs else 1.0
        per_agent_level[aid] = {"behaviour": ctx.agents[aid].get("behaviour"),
                                "is_adversary": aid in gt_bad,
                                "levels": d,
                                "verdict": agent_verdict(results, aid)[0]}

    # (c) per-level detection ownership
    detection_ownership = {}
    for aid in sorted(gt_bad):
        _, failed = agent_verdict(results, aid)
        flvls = sorted({by_id(cid).level for cid in failed})
        detection_ownership[aid] = {"behaviour": ctx.agents[aid].get("behaviour"),
                                    "flagged_by_levels": flvls,
                                    "controls": list(failed.keys())}

    # (d) level ablation
    level_ablation = []
    for lvl in levels:
        disabled = frozenset(cp.id for cp in by_level(lvl))
        flagged_ab = _flagged_set(results, ids, disabled=disabled)
        newly = sorted((gt_bad & flagged_full) - flagged_ab)
        acc = (fl.train_trajectory(prep, exclude=flagged_ab, profile=profile)["accuracy"]
               if newly else gov_acc)
        level_ablation.append({
            "level": lvl, "n_controls": len(disabled),
            "newly_undetected": newly,
            "undetected_behaviours": [ctx.agents[a].get("behaviour") for a in newly],
            "accuracy_under_ablation": round(acc, 4),
            "delta_vs_governance": round(acc - gov_acc, 4)})

    return {"seed": seed, "governance_accuracy": round(gov_acc, 4),
            "foundation_note": "LARGE is the system-wide gate; MEDIUM/SMALL score per agent",
            "rollup": rollup, "per_agent_level": per_agent_level,
            "detection_ownership": detection_ownership,
            "level_ablation": level_ablation}


def _e1e2_for_prep(prep: dict, profile: dict) -> dict:
    """naive/gov/oracle accuracy + detection + honest FPR for one prepared run.
    Shared by the B3 (non-IID) and deep-backbone experiments."""
    naive = fl.train_trajectory(prep, record=True, reset_ledger=True, profile=profile)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()
    gt_bad = set(profile) | ({config.REVOKED_AGENT}
                             if config.REVOKED_AGENT in ids else set())
    flagged = _flagged_set(results, ids)
    honest = set(ids) - gt_bad
    fpr = len(flagged & honest) / max(len(honest), 1)
    det = _conf(flagged, ids, gt_bad)
    gov = fl.train_trajectory(prep, exclude=flagged, profile=profile)["accuracy"]
    oracle = fl.train_trajectory(prep, exclude=gt_bad, profile=profile)["accuracy"]
    return {"naive": naive["accuracy"], "gov": gov, "oracle": oracle,
            "recall": det["recall"], "precision": det["precision"],
            "honest_fpr": round(fpr, 4)}


# ---------------------------------------------------------------------------
# B3 — non-IID severity sweep (Dirichlet alpha)
# ---------------------------------------------------------------------------
def run_noniid(alphas: list[float] | None = None,
               seeds: list[int] | None = None) -> dict:
    """Sweep data heterogeneity. Small alpha = extreme non-IID. Watches whether
    honest-but-skewed agents get false-flagged (the global-prior drift control
    S-DF-03 assumes IID)."""
    alphas = alphas or [10.0, 1.0, 0.5, 0.1]
    seeds = seeds or config.exp_seeds()[:2]
    profile = config.ADVERSARY_PROFILE
    rows = []
    for a in alphas:
        acc = {"naive": [], "gov": [], "oracle": [], "recall": [], "fpr": []}
        for s in seeds:
            prep = fl.prepare(s, partition="dirichlet", alpha=a)
            r = _e1e2_for_prep(prep, profile)
            acc["naive"].append(r["naive"]); acc["gov"].append(r["gov"])
            acc["oracle"].append(r["oracle"]); acc["recall"].append(r["recall"])
            acc["fpr"].append(r["honest_fpr"])
        rows.append({"alpha": a,
                     "naive_accuracy": _agg(acc["naive"]),
                     "governance_accuracy": _agg(acc["gov"]),
                     "oracle_accuracy": _agg(acc["oracle"]),
                     "adversary_recall": _agg(acc["recall"]),
                     "honest_false_positive_rate": _agg(acc["fpr"])})
    return {"seeds": seeds, "rows": rows}


def run_noniid_fix(alphas: list[float] | None = None,
                   seeds: list[int] | None = None) -> dict:
    """B3 remedy: compare the label-prior drift control (current S-DF-03) with a
    non-IID-aware FLTrust server-anchored direction control across the α sweep.
    Reports honest false-positive rate and directional-poison recall for each."""
    import numpy as np
    from . import fltrust
    alphas = alphas or [10.0, 1.0, 0.5, 0.1]
    seeds = seeds or config.exp_seeds()[:2]
    profile = config.ADVERSARY_PROFILE
    directional = {a for a, b in profile.items()
                   if b in ("label_flip", "collude_flip")}
    rows = []
    for al in alphas:
        agg = {"fpr_lp": [], "rec_lp": [], "fpr_ft": [], "rec_ft": []}
        for s in seeds:
            prep = fl.prepare(s, partition="dirichlet", alpha=al)
            naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                        profile=profile, return_updates=True,
                                        return_server_update=True)
            ctx = build_ctx(naive["signals"])
            ids = ctx.agent_ids()
            honest = set(ids) - set(profile) - {config.REVOKED_AGENT}
            # (a) label-prior detector = current S-DF-03 rule (mean label KL > 0.2)
            kl = {i: float(np.mean([r["label_kl"] for r in ctx.agents[i]["rounds"]]))
                  for i in ids}
            lp = {i for i in ids if kl[i] > 0.2}
            # (b) FLTrust server-anchored direction detector
            fts = fltrust.fltrust_scores(naive["updates"], naive["server_update"])
            ft = fltrust.fltrust_directional_flagged(fts, tau=0.0)

            def fpr(f):
                return len(f & honest) / max(len(honest), 1)

            def rec(f):
                return len(f & directional) / max(len(directional), 1)

            agg["fpr_lp"].append(fpr(lp)); agg["rec_lp"].append(rec(lp))
            agg["fpr_ft"].append(fpr(ft)); agg["rec_ft"].append(rec(ft))
        rows.append({"alpha": al,
                     "labelprior_honest_fpr": _agg(agg["fpr_lp"]),
                     "labelprior_recall": _agg(agg["rec_lp"]),
                     "fltrust_honest_fpr": _agg(agg["fpr_ft"]),
                     "fltrust_recall": _agg(agg["rec_ft"])})
    return {"seeds": seeds, "rows": rows}


# ---------------------------------------------------------------------------
# Deep-model backbone experiment (linear vs MLP)
# ---------------------------------------------------------------------------
def run_deep_backbone(backbones: list[str] | None = None,
                      seeds: list[int] | None = None) -> dict:
    """Repeat E1/E2 with a deep (MLP) backbone to show the governance mechanics
    hold beyond the linear model."""
    backbones = backbones or ["linear", "mlp"]
    seeds = seeds or config.exp_seeds()[:2]
    profile = config.ADVERSARY_PROFILE
    rows = []
    for bb in backbones:
        acc = {"naive": [], "gov": [], "oracle": [], "recall": [], "fpr": []}
        for s in seeds:
            prep = fl.prepare(s, backbone=bb)
            r = _e1e2_for_prep(prep, profile)
            acc["naive"].append(r["naive"]); acc["gov"].append(r["gov"])
            acc["oracle"].append(r["oracle"]); acc["recall"].append(r["recall"])
            acc["fpr"].append(r["honest_fpr"])
        rows.append({"backbone": bb,
                     "naive_accuracy": _agg(acc["naive"]),
                     "governance_accuracy": _agg(acc["gov"]),
                     "oracle_accuracy": _agg(acc["oracle"]),
                     "detection_recall": _agg(acc["recall"]),
                     "honest_false_positive_rate": _agg(acc["fpr"])})
    return {"seeds": seeds, "rows": rows}


# ---------------------------------------------------------------------------
# C9 — manifold / distribution post-training checks (DL/ML)
# ---------------------------------------------------------------------------
_UPDATE_ATTACKS = {"scale_attack", "stealth_scale", "label_flip", "collude_flip"}


def run_manifold(seed: int | None = None, profile: dict | None = None) -> dict:
    """Compare the update-*norm* detector (M-BC-01) with an update-*direction*
    manifold detector, and run a post-training output-distribution health check
    on the naive (poisoned) vs governance-gated global model."""
    import numpy as np
    from . import manifold as mf
    seed = config.EXP_BASE_SEED if seed is None else seed
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    prep = fl.prepare(seed)

    naive = fl.train_trajectory(prep, record=True, reset_ledger=True,
                                profile=profile, return_updates=True,
                                return_model=True)
    signals = naive["signals"]
    ctx = build_ctx(signals)
    results = evaluate_controls(ctx)
    ids = ctx.agent_ids()

    # detectors
    znorm = _anomaly_z(signals)                     # |z| of update norm
    mscores = mf.update_direction_scores(naive["updates"])
    norm_flagged = {i for i in ids if znorm[i] > config.ANOMALY_Z_TAU}
    man_flagged = mf.manifold_flagged(mscores, z_tau=2.0)
    combined = norm_flagged | man_flagged

    # ground truth: only the *update-manipulating* attackers are in scope here
    update_adv = {a for a, b in profile.items() if b in _UPDATE_ATTACKS}
    honest = set(ids) - set(profile) - {config.REVOKED_AGENT}

    def rec_fpr(flagged):
        tp = len(flagged & update_adv)
        rec = tp / max(len(update_adv), 1)
        fp = len(flagged & honest)
        fpr = fp / max(len(honest), 1)
        return round(rec, 3), round(fpr, 3)

    n_rec, n_fpr = rec_fpr(norm_flagged)
    m_rec, m_fpr = rec_fpr(man_flagged)
    c_rec, c_fpr = rec_fpr(combined)

    per_agent = []
    for i in ids:
        beh = ctx.agents[i].get("behaviour")
        caught = []
        if i in norm_flagged:
            caught.append("norm")
        if i in man_flagged:
            caught.append("manifold")
        per_agent.append({"agent": i, "behaviour": beh,
                          "norm_z": round(znorm[i], 2),
                          "cosine_z": mscores["cosine_z"][i],
                          "is_update_adversary": i in update_adv,
                          "caught_by": caught})

    # post-training distribution health: naive (poisoned) vs governance-gated
    flagged_all = _flagged_set(results, ids)
    gated = fl.train_trajectory(prep, exclude=flagged_all, profile=profile,
                                return_model=True)
    prior = prep["base_dist"]
    ncls = prep["n_classes"]
    health_naive = mf.model_output_health(naive["model"], prep["X_te"], ncls, prior)
    health_gated = mf.model_output_health(gated["model"], prep["X_te"], ncls, prior)

    return {
        "seed": seed,
        "detectors": {
            "norm_only": {"recall_update_attacks": n_rec, "fpr": n_fpr},
            "manifold_only": {"recall_update_attacks": m_rec, "fpr": m_fpr},
            "combined": {"recall_update_attacks": c_rec, "fpr": c_fpr},
        },
        "update_adversaries": sorted(update_adv),
        "per_agent": per_agent,
        "post_training_health": {"naive_model": health_naive,
                                 "governance_gated_model": health_gated},
    }
