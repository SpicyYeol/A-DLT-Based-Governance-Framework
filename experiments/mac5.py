# -*- coding: utf-8 -*-
"""Marginal accuracy contribution MAC(a) = A(N) - A(N\\{a}) over FIVE seeds,
for consistency with every other primary table (the shipped helper used three).
Usage: python mac5.py <out.json>
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from pathlib import Path

TB = Path(os.environ.get("TRUST_TESTBED",
                    Path(__file__).resolve().parents[1]))
os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

from trust import config                                  # noqa: E402
from trust import fl_simulation as fl                      # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
LABEL = {0: "honest", 1: "label flipping", 2: "honest", 3: "scaling",
         4: "honest", 5: "replay/stale", 6: "tampering",
         7: "revoked credential"}


def main(out_path):
    profile = config.ADVERSARY_PROFILE
    gt = set(profile) | {config.REVOKED_AGENT}
    per = {}
    for s in SEEDS:
        prep = fl.prepare(s)
        base = fl.train_trajectory(prep, profile=profile)["accuracy"]
        for sh in prep["shards"]:
            a = sh["agent_id"]
            wo = fl.train_trajectory(prep, exclude={a}, profile=profile)["accuracy"]
            per.setdefault(a, []).append(base - wo)
        print("  seed %d done (base %.4f)" % (s, base), flush=True)

    rows = []
    for a, v in sorted(per.items()):
        rows.append({"agent": a, "behaviour": LABEL.get(a, "?"),
                     "is_adversary": a in gt,
                     "mac_mean": round(st.mean(v), 4),
                     "mac_sd": round(st.pstdev(v), 4),
                     "per_seed": [round(x, 4) for x in v]})
    hon = [r for r in rows if not r["is_adversary"]]
    print()
    print("%-6s %-20s %-6s %18s" % ("agent", "behaviour", "adv", "MAC"))
    for r in rows:
        print("%-6d %-20s %-6s %8.4f +- %.4f"
              % (r["agent"], r["behaviour"], "yes" if r["is_adversary"] else "no",
                 r["mac_mean"], r["mac_sd"]))
    print()
    print("honest MAC range: %.4f to %.4f"
          % (min(r["mac_mean"] for r in hon), max(r["mac_mean"] for r in hon)))
    print("sum of MAC over all participants: %.4f"
          % sum(r["mac_mean"] for r in rows))
    Path(out_path).write_text(
        json.dumps({"seeds": SEEDS, "image_cap": config.IMAGE_SAMPLE_CAP,
                    "rows": rows}, indent=1), encoding="utf-8")
    print("saved:", out_path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
