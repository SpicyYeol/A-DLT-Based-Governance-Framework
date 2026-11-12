# -*- coding: utf-8 -*-
"""Re-measure one (N, seed) point of the scalability sweep and refresh the
summary in place.

Used when a single measurement has to be repeated, for instance because the
host was busy while it was taken. The point is replaced in
reports_rev1/rev1_scale.json and the per-N summary is recomputed from the
rows that file then holds.

Run:  TRUST_DATA_ROOT=<dir> python scripts/rev1_scale_fix.py <N> <seed>
"""
import json, os, sys, statistics as st, importlib.util


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: rev1_scale_fix.py <N> <seed>")
        return 2
    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location("rs", os.path.join(here, "rev1_scale.py"))
    rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    p = rs.OUT / "rev1_scale.json"
    D = json.load(open(p))
    new = rs.measure(n, seed)
    print(json.dumps(new))
    D["rows"] = [r for r in D["rows"] if not (r["N"] == n and r["seed"] == seed)] + [new]
    D["rows"].sort(key=lambda r: (r["N"], r["seed"]))
    summary = []
    for N in sorted({r["N"] for r in D["rows"]}):
        rr = [r for r in D["rows"] if r["N"] == N]
        summary.append({"N": N, "blocks": rr[0]["blocks"], "ledger_kb": rr[0]["ledger_bytes"] / 1024,
            "t_train_and_record_s": st.mean(r["t_train_and_record_s"] for r in rr),
            "t_governance_ms": 1e3 * st.mean(r["t_governance_s"] for r in rr),
            # rev1_scale.py writes this one; without it, repairing a single
            # point would drop the dispersion for every N in the file
            "t_governance_ms_sd": 1e3 * st.pstdev([r["t_governance_s"] for r in rr]),
            "t_governance_ms_median": 1e3 * st.median(r["t_governance_s"] for r in rr),
            "t_chain_ms": 1e3 * st.mean(r["t_chain_verify_s"] for r in rr),
            "all_chain_ok": all(r["chain_ok"] for r in rr),
            **{k: sum(r[k] for r in rr) for k in ("TP", "FP", "FN", "TN")}})
    D["summary"] = summary
    json.dump(D, open(p, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
