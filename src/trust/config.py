"""Runtime configuration for the trustworthiness testbed.

All paths default under a shared data directory so Airflow tasks in separate
containers exchange artifacts as JSON. Override with env vars.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.getenv("TRUST_DATA_ROOT", "/opt/airflow/data"))
LEDGER_DIR = DATA_ROOT / "ledger"
REPORTS_DIR = DATA_ROOT / "reports"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"   # intermediate signals between DAG tasks
DATASET_CACHE = DATA_ROOT / "dataset_cache"

for _d in (LEDGER_DIR, REPORTS_DIR, ARTIFACTS_DIR, DATASET_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# --- testbed scenario knobs -------------------------------------------------
NUM_AGENTS = int(os.getenv("TRUST_NUM_AGENTS", "8"))
NUM_ROUNDS = int(os.getenv("TRUST_FL_ROUNDS", "3"))
RANDOM_SEED = int(os.getenv("TRUST_SEED", "42"))

# Data volume caps keep a run fast inside a container.
IMAGE_SAMPLE_CAP = int(os.getenv("TRUST_IMAGE_CAP", "8000"))
# Image dataset for the FL branch: mnist | fashion_mnist  (C8 varies this).
IMAGE_DATASET = os.getenv("TRUST_IMAGE_DATASET", "mnist")
TABULAR_SAMPLE_CAP = int(os.getenv("TRUST_TABULAR_CAP", "20000"))

# Freshness SLA (seconds) for SMALL / Data Freshness checks.
FRESHNESS_SLA_SEC = int(os.getenv("TRUST_FRESHNESS_SLA", "300"))

# HMAC secret standing in for each agent's signing key material.
SIGNING_SCHEME = "HMAC-SHA256"

# Robust modified-z outlier threshold for the update-anomaly control (M-BC-01).
# Clause-8 operating point; C5 sweeps this.
ANOMALY_Z_TAU = float(os.getenv("TRUST_ANOMALY_Z_TAU", "3.0"))

# Injected adversary profile: which simulated agents misbehave and how.
# Kept deterministic so runs are reproducible and gradable.
ADVERSARY_PROFILE = {
    # agent_id : behaviour
    1: "label_flip",       # data poisoning  -> update anomaly + drift + accuracy drop
    3: "scale_attack",     # gradient/scale poisoning -> large update norm
    5: "stale",            # lazy agent, replays old timestamps -> freshness fail
    6: "tamper",           # flips payload after hashing -> hash mismatch
    # remaining agents are honest
}

# Agent whose credential we revoke to exercise identity-lifecycle control.
REVOKED_AGENT = int(os.getenv("TRUST_REVOKED_AGENT", "7"))

# --- experiment settings ---------------------------------------------------
# Number of random seeds for the multi-seed variance experiment (C1). Seeds are
# BASE_SEED, BASE_SEED+1, ...  Keep small for the DAG; raise for a paper table.
EXP_NUM_SEEDS = int(os.getenv("TRUST_EXP_SEEDS", "5"))
EXP_BASE_SEED = int(os.getenv("TRUST_EXP_BASE_SEED", "42"))

def exp_seeds() -> list[int]:
    return [EXP_BASE_SEED + i for i in range(EXP_NUM_SEEDS)]

# --- C6 stealth / collusion threat profile ---------------------------------
# Low-magnitude scale factor chosen to slip under the |z|<=3 anomaly threshold.
STEALTH_SCALE_FACTOR = float(os.getenv("TRUST_STEALTH_FACTOR", "1.3"))

# A harder profile: a stealthy (small) model-poisoner plus TWO colluding
# label-flip agents that flip to the SAME target class (coordinated), so their
# poisoned updates look mutually consistent rather than like lone outliers.
STEALTH_PROFILE = {
    1: "collude_flip",
    2: "collude_flip",
    3: "stealth_scale",
    6: "tamper",       # keep one overt integrity attack as a control
}

