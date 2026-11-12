# -*- coding: utf-8 -*-
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
      preventing, which is the risk of not verifying untrustworthy agents at
      all. Two harm axes are reported:
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

The experiments are split across the modules of this package by family, and
everything is re-exported here, so `experiments.run_all()` and every other
call site is unchanged.
"""
from __future__ import annotations

from . import _core as _core  # noqa: F401
from . import baselines as baselines  # noqa: F401
from . import stealth as stealth  # noqa: F401
from . import thresholds as thresholds  # noqa: F401
from . import fractions as fractions  # noqa: F401
from . import ablation as ablation  # noqa: F401
from . import scalability as scalability  # noqa: F401
from . import generalization as generalization  # noqa: F401
from . import noniid as noniid  # noqa: F401
from . import backbone as backbone  # noqa: F401
from . import manifold_checks as manifold_checks  # noqa: F401

from ._core import *  # noqa: F401,F403
from .baselines import *  # noqa: F401,F403
from .stealth import *  # noqa: F401,F403
from .thresholds import *  # noqa: F401,F403
from .fractions import *  # noqa: F401,F403
from .ablation import *  # noqa: F401,F403
from .scalability import *  # noqa: F401,F403
from .generalization import *  # noqa: F401,F403
from .noniid import *  # noqa: F401,F403
from .backbone import *  # noqa: F401,F403
from .manifold_checks import *  # noqa: F401,F403


# The star imports above skip names that begin with an underscore, so the
# helpers that the drivers and trust.ml_eval import from this module by name
# are re-exported explicitly. They are internal to the framework, but they were
# importable before the split and the call sites expect them.
from ._core import (  # noqa: F401
    _agg,
    _conf,
    _flagged_set,
    _untrusted_only,
)
from .thresholds import _anomaly_z  # noqa: F401
from .fractions import _ATTACK_CYCLE, _ATTACK_ORDER  # noqa: F401
from .scalability import _measure_scale  # noqa: F401
from .generalization import _e1e2_for_prep  # noqa: F401
from .manifold_checks import _UPDATE_ATTACKS  # noqa: F401
