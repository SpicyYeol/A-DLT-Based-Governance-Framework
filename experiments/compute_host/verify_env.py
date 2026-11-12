# -*- coding: utf-8 -*-
"""Verify that the server environment matches the pinned configuration and that
MNIST really loads from the shipped OpenML cache.

The testbed's image loader silently falls back to sklearn's bundled 8x8
``load_digits`` on any fetch failure, which would produce numbers that look
plausible but come from a different dataset. This script makes that failure
loud instead.

Exit status is non-zero if anything does not match.
"""
import os
import sys
from pathlib import Path

PINNED = {
    "python": "3.12",          # prefix match
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "sklearn": "1.5.2",
    "pandas": "2.2.3",
}

ROOT = Path(__file__).resolve().parents[2]
TB = Path(os.environ.get("TRUST_TESTBED", str(ROOT)))

os.environ.setdefault("TRUST_DATA_ROOT", str(TB / "data" / "exp"))
os.environ.setdefault("TRUST_IMAGE_CAP", "4000")
sys.path.insert(0, str(TB / "src"))

bad = []

import numpy
import scipy
import sklearn
import pandas

got = {
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "pandas": pandas.__version__,
}
for k, want in PINNED.items():
    ok = got[k].startswith(want) if k == "python" else got[k] == want
    print("%-8s %-10s %s" % (k, got[k], "ok" if ok else "EXPECTED " + want))
    if not ok:
        bad.append("%s=%s (want %s)" % (k, got[k], want))

print()
print("BLAS threads env:",
      {v: os.environ.get(v) for v in
       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

print()
print("--- dataset ---")
from trust import config
from trust.datasets import load_image_dataset

print("DATA_ROOT     ", config.DATA_ROOT)
print("DATASET_CACHE ", config.DATASET_CACHE)
print("IMAGE_CAP     ", config.IMAGE_SAMPLE_CAP)

ds = load_image_dataset("mnist")
print("name          ", ds["name"])
print("X shape       ", ds["X"].shape, ds["X"].dtype)
print("n_classes     ", ds["n_classes"])
print("X range       ", float(ds["X"].min()), float(ds["X"].max()))

if "fallback" in ds["name"]:
    bad.append("image loader fell back to bundled digits: %s" % ds["name"])
if ds["X"].shape[1] != 784:
    bad.append("feature count %d, expected 784 (MNIST)" % ds["X"].shape[1])
if ds["X"].shape[0] != config.IMAGE_SAMPLE_CAP:
    bad.append("sample count %d, expected cap %d"
               % (ds["X"].shape[0], config.IMAGE_SAMPLE_CAP))

print()
print("--- control taxonomy ---")
from trust.taxonomy import ALL_CONTROLS
print("control points:", len(ALL_CONTROLS))
if len(ALL_CONTROLS) != 36:
    bad.append("taxonomy has %d control points, expected 36" % len(ALL_CONTROLS))

print()
if bad:
    print("VERIFICATION FAILED:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("VERIFICATION OK")
