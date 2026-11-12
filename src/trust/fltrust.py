"""Non-IID-aware drift control (FLTrust-style server-anchored trust).

B3 showed the label-prior drift control (S-DF-03) over-flags honest-but-skewed
agents under extreme non-IID, because it compares each agent's label
distribution to the *global* prior. This module replaces that baseline with a
server-anchored **direction** test (Cao et al., FLTrust 2021):

* the server holds a small clean, balanced root set and computes its own update
  direction g0 from the current global model;
* each agent i is scored by ReLU(cos(g_i, g0)) — how well its update *direction*
  aligns with the clean reference.

Why it is non-IID robust: a legitimately skewed honest agent still moves the
model in a direction consistent with the clean root (cos > 0), whereas a
label-flip / backdoor agent pushes the opposite way (cos <= 0). The reference is
a gradient direction, not a label histogram, so natural label skew no longer
looks like an attack.
"""

from __future__ import annotations

import numpy as np


def fltrust_scores(agent_updates: dict, server_update) -> dict:
    """agent_updates: {agent_id: flattened update vector}; server_update: the
    clean root update. Returns per-agent cosine, ReLU-cosine and norm ratio."""
    g0 = np.asarray(server_update, dtype=float)
    n0 = float(np.linalg.norm(g0)) or 1e-12
    out = {}
    for aid, v in agent_updates.items():
        gi = np.asarray(v, dtype=float)
        ni = float(np.linalg.norm(gi)) or 1e-12
        cos = float(np.dot(gi, g0) / (ni * n0))
        out[aid] = {"cos": round(cos, 4), "relu_cos": round(max(0.0, cos), 4),
                    "norm_ratio": round(ni / n0, 4)}
    return out


def fltrust_directional_flagged(scores: dict, tau: float = 0.1) -> set:
    """Agents whose update direction does not align with the clean reference
    (ReLU-cosine <= tau) — directional poisoning, robust to non-IID skew."""
    return {aid for aid, s in scores.items() if s["relu_cos"] <= tau}
