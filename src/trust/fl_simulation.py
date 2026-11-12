"""Federated-learning simulation that produces the raw trust signals.

N agents each hold a shard of the open image dataset and train a local model
each round (FedAvg over linear-model weights). A deterministic subset misbehaves
per ``config.ADVERSARY_PROFILE``:

    label_flip   -> data poisoning: local accuracy drops, label KL rises
    scale_attack -> model/gradient poisoning: huge update norm
    stale        -> replays old timestamps (SMALL / Data Freshness)
    tamper       -> payload altered after hashing (SMALL / Message Integrity)

Every agent also submits a per-round *data packet* to the DLT trust registry so
the traceability / integrity / freshness controls have real ledger evidence.

The core primitive is :func:`train_trajectory`, which trains one FedAvg
trajectory while *excluding* an arbitrary set of agents from aggregation. That
single primitive powers every accuracy comparison the experiments need:

    naive       = train_trajectory(exclude = {})                # no gating
    oracle      = train_trajectory(exclude = ground-truth bad)  # upper bound
    governance  = train_trajectory(exclude = verdict==Untrusted) # closed-loop
    ablation-X  = train_trajectory(exclude = verdict without control X)
"""

from __future__ import annotations

import time
import numpy as np

from . import config
from .datasets import load_image_dataset, partition_for_agents, partition_dirichlet
from .ledger import Ledger, sign, sha256_hex

EPOCHS = 4  # local passes per round -> healthier accuracy


def _principal(agent_id: int) -> str:
    return f"org-{agent_id % 3}"


def _label_kl(y_local: np.ndarray, n_classes: int, base: np.ndarray) -> float:
    counts = np.bincount(y_local, minlength=n_classes).astype(float)
    p = counts / max(counts.sum(), 1)
    eps = 1e-9
    return float(np.sum(np.where(p > 0, p * np.log((p + eps) / (base + eps)), 0.0)))


def prepare(seed: int | None = None, dataset: str | None = None,
            partition: str = "iid", alpha: float = 1.0,
            backbone: str = "linear") -> dict:
    """Load + split + partition the image dataset. Deterministic in ``seed``.

    ``dataset`` selects mnist | fashion_mnist (C8); ``partition`` selects
    iid | dirichlet with heterogeneity ``alpha`` (B3); ``backbone`` selects
    linear | mlp (deep-model experiment)."""
    from sklearn.model_selection import train_test_split
    seed = config.RANDOM_SEED if seed is None else seed
    ds = load_image_dataset(dataset)
    # models need one float dtype end-to-end; we set weights by hand for FedAvg,
    # so pin everything to float64 to avoid 32/64 dataset mismatches.
    X, y, n_classes = ds["X"].astype(np.float64), ds["y"], ds["n_classes"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    if partition == "dirichlet":
        shards = partition_dirichlet(X_tr, y_tr, config.NUM_AGENTS, alpha, seed,
                                     n_classes)
    else:
        shards = partition_for_agents(X_tr, y_tr, config.NUM_AGENTS, seed)
    base_dist = np.bincount(y_tr, minlength=n_classes).astype(float)
    base_dist /= base_dist.sum()
    return {
        "dataset_name": ds["name"], "seed": seed, "n_classes": n_classes,
        "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te,
        "shards": shards, "base_dist": base_dist,
        "partition": partition, "alpha": alpha, "backbone": backbone,
    }


def _new_model(backbone: str, seed: int):
    if backbone == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64,), activation="relu",
                             solver="adam", learning_rate_init=0.02,
                             max_iter=1, random_state=seed)
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(loss="log_loss", random_state=seed, max_iter=5,
                         tol=None, alpha=1e-4)


def _get_w(m):
    """Return (coefs, intercepts) as lists of arrays — one entry per layer for
    an MLP, a single entry for a linear model."""
    if hasattr(m, "coefs_"):                  # MLPClassifier
        return [c.copy() for c in m.coefs_], [b.copy() for b in m.intercepts_]
    return [m.coef_.copy()], [m.intercept_.copy()]


def _set_w(m, coefs, ints):
    if hasattr(m, "coefs_"):
        m.coefs_ = [c.copy() for c in coefs]
        m.intercepts_ = [b.copy() for b in ints]
    else:
        m.coef_ = coefs[0].copy()
        m.intercept_ = ints[0].copy()


def train_trajectory(prep: dict, exclude: set[int] | frozenset = frozenset(),
                     *, record: bool = False, reset_ledger: bool = False,
                     profile: dict | None = None, aggregator=None,
                     return_updates: bool = False, return_model: bool = False,
                     return_server_update: bool = False) -> dict:
    """Train one FedAvg trajectory, excluding ``exclude`` agents from aggregation.

    ``record=True`` also writes the DLT ledger and collects per-agent signals
    (used for the single reference/naive pass). Adversary behaviour is always
    applied to *included* agents; excluded agents simply do not contribute.

    ``profile`` overrides the adversary profile (for the C6 stealth experiment);
    ``aggregator`` overrides FedAvg with a robust rule (for the C3 baselines).
    """
    seed = prep["seed"]
    classes = np.arange(prep["n_classes"])
    X_tr, y_tr, X_te, y_te = prep["X_tr"], prep["y_tr"], prep["X_te"], prep["y_te"]
    shards, base_dist = prep["shards"], prep["base_dist"]
    profile = config.ADVERSARY_PROFILE if profile is None else profile
    rng = np.random.RandomState(seed)

    backbone = prep.get("backbone", "linear")

    def _init_model():
        m = _new_model(backbone, seed)
        m.partial_fit(X_tr[:256], y_tr[:256], classes=classes)
        return m

    def _local_fit(gW, Xl, yl):
        m = _new_model(backbone, seed)
        m.partial_fit(Xl[:256], yl[:256], classes=classes)
        _set_w(m, gW[0], gW[1])
        for _ in range(EPOCHS):
            order = rng.permutation(len(Xl))
            m.partial_fit(Xl[order], yl[order])
        return m

    def _wavg(Ws, weights):
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        n_c, n_i = len(Ws[0][0]), len(Ws[0][1])
        coefs = [sum(w[k] * Ws[k][0][l] for k in range(len(Ws))) for l in range(n_c)]
        ints = [sum(w[k] * Ws[k][1][l] for k in range(len(Ws))) for l in range(n_i)]
        return coefs, ints

    ledger = None
    if record:
        ledger = Ledger()
        if reset_ledger:
            ledger.reset()

    gm = _init_model()

    agents = {sh["agent_id"]: {
        "agent_id": sh["agent_id"], "principal": _principal(sh["agent_id"]),
        "behaviour": profile.get(sh["agent_id"], "honest"),
        "is_adversary": sh["agent_id"] in profile,
        "n_samples": int(len(sh["y"])), "rounds": []}
        for sh in shards}

    upd_accum = {sh["agent_id"]: [] for sh in shards} if return_updates else None

    # FLTrust clean root set (balanced) + server-update accumulator
    srv_accum = None
    if return_server_update:
        srv_accum = []
        root_idx = []
        for c in range(prep["n_classes"]):
            ci = np.where(y_tr == c)[0]
            root_idx.extend(ci[:10].tolist())
        X_root, y_root = X_tr[np.array(root_idx)], y_tr[np.array(root_idx)]

    now = time.time()
    for rnd in range(config.NUM_ROUNDS):
        contribs = []
        gW = _get_w(gm)                                  # (coefs list, ints list)

        if return_server_update:                         # clean server direction
            ms = _local_fit(gW, X_root, y_root)
            sW = _get_w(ms)
            sd_c = [lc - gc for lc, gc in zip(sW[0], gW[0])]
            sd_i = [li - gi for li, gi in zip(sW[1], gW[1])]
            srv_accum.append(np.concatenate(
                [a.ravel() for a in sd_c] + [a.ravel() for a in sd_i]))

        for sh in shards:
            aid = sh["agent_id"]
            beh = profile.get(aid, "honest")
            Xl, yl = sh["X"].copy(), sh["y"].copy()

            if beh in ("label_flip", "collude_flip"):   # data poisoning + drift
                yl = yl.copy()
                yl[rng.rand(len(yl)) < 0.7] = 0          # colluders share target 0

            m = _local_fit(gW, Xl, yl)
            lW = _get_w(m)
            d_coefs = [lc - gc for lc, gc in zip(lW[0], gW[0])]
            d_ints = [li - gi for li, gi in zip(lW[1], gW[1])]
            if beh == "scale_attack":                   # overt model poisoning
                d_coefs = [x * 12.0 for x in d_coefs]; d_ints = [x * 12.0 for x in d_ints]
            elif beh == "stealth_scale":                # low-magnitude evasion
                f = config.STEALTH_SCALE_FACTOR
                d_coefs = [x * f for x in d_coefs]; d_ints = [x * f for x in d_ints]

            new_coefs = [gc + dc for gc, dc in zip(gW[0], d_coefs)]
            new_ints = [gi + di for gi, di in zip(gW[1], d_ints)]

            # contribute to aggregation unless excluded
            if aid not in exclude:
                contribs.append({"W": (new_coefs, new_ints), "weight": len(yl),
                                 "aid": aid, "coef": new_coefs[0], "int": new_ints[0]})

            if return_updates:              # capture the raw update *direction*
                upd_accum[aid].append(np.concatenate(
                    [a.ravel() for a in d_coefs] + [a.ravel() for a in d_ints]))

            if record:
                update_norm = float(sum(np.linalg.norm(a) for a in d_coefs)
                                    + sum(np.linalg.norm(a) for a in d_ints))
                local_acc = float(m.score(X_te, y_te))
                kl = _label_kl(yl, prep["n_classes"], base_dist)
                payload = f"agent{aid}-round{rnd}-n{len(yl)}".encode()
                claimed_hash = signature = None
                ts = now
                if beh == "stale":
                    ts = now - config.FRESHNESS_SLA_SEC * 5
                if beh == "tamper":
                    claimed_hash = sha256_hex(payload + b"-tampered")
                    signature = sign(aid, sha256_hex(payload))
                blk = ledger.append(
                    agent_id=aid, principal=_principal(aid), tx_type="data_packet",
                    payload=payload, claimed_hash=claimed_hash, signature=signature,
                    timestamp=ts, ingest_time=now,
                    meta={"round": rnd, "behaviour": beh,
                          "local_acc": round(local_acc, 4),
                          "update_norm": round(update_norm, 6),
                          "label_kl": round(kl, 6), "logged": True,
                          "model_version": rnd + 1})
                agents[aid]["rounds"].append({
                    "round": rnd, "update_norm": update_norm,
                    "local_acc": local_acc, "label_kl": kl,
                    "block_index": blk.index, "timestamp": ts,
                    "ingest_time": now, "model_version": rnd + 1})

        if contribs:  # at least one agent contributed this round
            if aggregator is None:
                cc, ii = _wavg([c["W"] for c in contribs],
                               [c["weight"] for c in contribs])
                _set_w(gm, cc, ii)
            else:                       # robust aggregators (C3) — linear only
                coef, intc = aggregator(contribs)
                _set_w(gm, [coef], [intc])

    acc = float(gm.score(X_te, y_te))
    out = {"accuracy": acc, "excluded": sorted(exclude)}
    if return_updates:
        out["updates"] = {aid: np.mean(vs, axis=0).tolist()
                          for aid, vs in upd_accum.items() if vs}
    if return_server_update:
        out["server_update"] = np.mean(srv_accum, axis=0).tolist()
    if return_model:
        out["model"] = gm
    if record:
        out["signals"] = {
            "dataset": prep["dataset_name"], "num_agents": config.NUM_AGENTS,
            "num_rounds": config.NUM_ROUNDS, "global_accuracy": acc,
            "revoked_agent": config.REVOKED_AGENT,
            "signing_scheme": config.SIGNING_SCHEME, "agents": agents,
            "seed": seed, "generated_at": now}
    return out


def run_simulation(reset_ledger: bool = True, seed: int | None = None) -> dict:
    """Reference (naive) pass used by the main governance DAG.

    Trains the naive all-agent model (records the ledger + signals) and, for the
    demo summary, an oracle-gated model that excludes ground-truth-bad agents.
    """
    prep = prepare(seed)
    naive = train_trajectory(prep, exclude=frozenset(), record=True,
                             reset_ledger=reset_ledger)
    signals = naive["signals"]
    bad = set(config.ADVERSARY_PROFILE) | {config.REVOKED_AGENT}
    oracle = train_trajectory(prep, exclude=bad)
    signals["gated_accuracy"] = oracle["accuracy"]
    signals["accuracy_uplift_from_gating"] = round(
        oracle["accuracy"] - naive["accuracy"], 4)
    return signals
