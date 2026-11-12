# -*- coding: utf-8 -*-
"""Summarise rev1_ledger.json into paper-ready numbers (with exact 95% CIs)."""
import json, sys, os
from collections import defaultdict
from scipy.stats import beta
root = sys.argv[1]
D = json.load(open(os.path.join(root, "reports_rev1", "rev1_ledger.json")))

def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return f"{k}/{n} ({k/n:.3f}) [{lo:.3f}, {hi:.3f}]"

print("runtime verdicts identical across C0/C1:", all(d["runtime_verdicts_identical"] for d in D))
print("runtime verdicts seed42:", D[0]["runtime_verdicts"])
print("bytes C0/C1/C2:", D[0]["bytes_C0"], D[0]["bytes_C1"], D[0]["bytes_C2"], "blocks", D[0]["n_blocks"])
print("verify time (ms):", [round(d["t_verify_s"] * 1e3, 3) for d in D])

print("\n== targeted (counts over seeds) ==")
agg = defaultdict(lambda: defaultdict(int))
for d in D:
    for r in d["targeted"]:
        key = (r["scenario"], r["strength"])
        agg[key]["n"] += 1
        for c in ("C0", "C1", "C2_k1", "C2_k2", "C2_kR"):
            x = r[c]
            agg[key][c + ".alarm"] += x["alarm"]
            agg[key][c + ".changed"] += x["changed"]
            agg[key][c + ".silent"] += (x["changed"] and not x["alarm"])
            agg[key][c + ".recovered"] += x.get("recovered", False)
            if "status" in x:
                agg[key][c + ".status:" + x["status"]] += 1
for k, v in agg.items():
    print(k, dict(v))

print("\n== random single-record manipulations ==")
for strength in ("edit", "rewrite"):
    rows = [r for d in D for r in d["random"] if r["strength"] == strength]
    n = len(rows)
    for c in ("C0", "C1", "C2_k1"):
        al = sum(r[c]["alarm"] for r in rows)
        ch = sum(r[c]["changed"] for r in rows)
        si = sum(r[c]["changed"] and not r[c]["alarm"] for r in rows)
        rc = sum(r[c].get("recovered", False) for r in rows)
        print(f"{strength:8s} {c:6s} alarm {cp(al,n)} | verdict changed {cp(ch,n)} | silent {cp(si,n)} | recovered {rc}/{n}")
    miss = [r["field"] for r in rows if not r["C1"]["alarm"]]
    print("   C1 undetected fields:", miss[:20])
    # which fields change verdicts silently under C0
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["field"]][0] += 1; by[r["field"]][1] += r["C0"]["changed"]
    print("   C0 verdict-changing by field:", {f: f"{b}/{a}" for f, (a, b) in sorted(by.items())})
