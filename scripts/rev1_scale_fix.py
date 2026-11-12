# Re-measure one (N, seed) point of rev1_scale and refresh the summary.
import json, os, sys, statistics as st, importlib.util
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
        "t_governance_ms_median": 1e3 * st.median(r["t_governance_s"] for r in rr),
        "t_chain_ms": 1e3 * st.mean(r["t_chain_verify_s"] for r in rr),
        "all_chain_ok": all(r["chain_ok"] for r in rr),
        **{k: sum(r[k] for r in rr) for k in ("TP", "FP", "FN", "TN")}})
D["summary"] = summary
json.dump(D, open(p, "w"), indent=1)
