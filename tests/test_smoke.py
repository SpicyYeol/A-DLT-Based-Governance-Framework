# -*- coding: utf-8 -*-
"""Fast checks that the installation works and the framework matches the
structure the paper describes. No model is trained, so this finishes in
seconds.

    python tests/test_smoke.py        # no pytest needed
    pytest tests/test_smoke.py        # or with pytest, if you have it

What is checked is what the paper claims about the specification itself: the
3-12-36 shape, the split between measured and attestation controls, the three
severity classes, that the hash chain detects a rewritten record, and that
importing a runner does not start an experiment.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TRUST_DATA_ROOT", str(ROOT / "data" / "exp"))

from trust import config                                   # noqa: E402
from trust.ledger import Ledger, sha256_hex                # noqa: E402
from trust.taxonomy import ALL_CONTROLS, CATEGORIES, by_level, by_id  # noqa: E402

LEVELS = ("LARGE", "MEDIUM", "SMALL")       # Structural, Behavioral, Granular


def test_taxonomy_is_3_12_36() -> None:
    """Three levels, 12 categories, 36 control points, 12 per level."""
    assert len(ALL_CONTROLS) == 36, len(ALL_CONTROLS)
    assert len(CATEGORIES) == 12, len(CATEGORIES)
    assert sorted({c.level for c in ALL_CONTROLS}) == sorted(LEVELS)
    for level in LEVELS:
        assert len(by_level(level)) == 12, (level, len(by_level(level)))


def test_control_ids_are_unique_and_resolvable() -> None:
    ids = [c.id for c in ALL_CONTROLS]
    assert len(set(ids)) == 36, "duplicate control id"
    for cid in ids:
        assert by_id(cid).id == cid


def test_measured_and_attestation_split() -> None:
    """28 controls are measured from runtime evidence; 8 rest on attestation."""
    methods = Counter(c.method for c in ALL_CONTROLS)
    assert methods["attestation"] == 8, methods
    assert methods["data"] == 28, methods


def test_three_severity_classes() -> None:
    """Every control carries Reject, Review or Info severity."""
    severities = Counter(c.severity for c in ALL_CONTROLS)
    assert set(severities) == {"reject", "review", "info"}, severities
    assert sum(severities.values()) == 36


def test_controls_the_paper_names_exist() -> None:
    """The seven control points the injected threats exercise."""
    fired = ["L-IA-03", "M-BC-01", "M-CV-02", "M-CV-03",
             "S-DF-02", "S-DF-03", "S-MI-02"]
    for cid in fired:
        assert by_id(cid) is not None, cid
    # the control that reads a label distribution carries Review, not Reject,
    # which is why a label-prior deviation quarantines rather than excludes
    assert by_id("S-DF-03").severity == "review"


def test_ledger_detects_a_rewritten_record() -> None:
    """The hash chain is continuous when appended to, and a record edited in
    place no longer recomputes to its stored block hash."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ledger.jsonl"
        ledger = Ledger(path)
        ledger.reset()
        for i in range(4):
            ledger.append(agent_id=i, principal="org-%d" % i,
                           tx_type="model_update", payload=b"payload-%d" % i)

        clean = ledger.verify_chain()
        assert clean["blocks"] == 4, clean
        assert clean["continuity"] == 1.0, clean
        assert clean["block_hash_integrity"] == 1.0, clean

        lines = path.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace('"payload-1"', '"payload-tampered"')
        lines[1] = lines[1].replace(sha256_hex(b"payload-1"),
                                    sha256_hex(b"payload-tampered"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        tampered = Ledger(path).verify_chain()
        assert tampered["block_hash_integrity"] < 1.0, (
            "an edited record was not detected: %s" % tampered)


def test_configuration_is_readable() -> None:
    """The image budget is the one thing worth checking before a run: the
    default is 8000, every reported table uses 4000, and the drivers set it."""
    assert config.NUM_AGENTS >= 1
    assert config.IMAGE_SAMPLE_CAP > 0
    assert config.ANOMALY_Z_TAU > 0


def test_no_module_uses_a_name_it_never_binds() -> None:
    """Catch the one way splitting a module into a package breaks silently: a
    function keeps calling a helper that landed in a sibling module, and nothing
    complains until that function is called, often after minutes of training.

    For every module of the two packages, collect what it binds (imports,
    assignments, definitions, parameters) and what it loads, and require that
    nothing is loaded without being bound.
    """
    import ast
    import builtins

    known = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__package__", "__spec__",
        "__builtins__",
    }

    def bound_and_loaded(tree: ast.AST) -> tuple[set[str], dict[str, int]]:
        bound: set[str] = set()
        loaded: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    bound.add("*" if a.name == "*" else (a.asname or a.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound.update(node.names)
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    bound.add(node.id)
                else:
                    loaded.setdefault(node.id, node.lineno)
        return bound, loaded

    checked = 0
    for package in ("experiments", "report"):
        for path in sorted((ROOT / "src" / "trust" / package).glob("*.py")):
            if path.name == "__init__.py":
                continue            # the re-export hub binds through import *
            bound, loaded = bound_and_loaded(
                ast.parse(path.read_text(encoding="utf-8"), str(path)))
            if "*" in bound:
                continue
            missing = {n: ln for n, ln in loaded.items()
                       if n not in bound and n not in known}
            assert not missing, (path.name, missing)
            checked += 1
    assert checked >= 14, checked


def test_runners_do_not_run_on_import() -> None:
    """Importing a runner must define its work, not do it. Every script under
    scripts/ keeps its work inside a main() behind a __name__ guard, so that
    importing one starts no experiment and produces no output.

    Importing a runner does still create the framework's own data directories,
    because trust.config makes them when it is imported; what is checked here
    is that the runner itself does nothing."""
    import contextlib
    import importlib.util

    scripts = sorted((ROOT / "scripts").glob("*.py"))
    assert len(scripts) >= 9, len(scripts)
    for path in scripts:
        spec = importlib.util.spec_from_file_location(
            "runner_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(module)
        assert not buf.getvalue().strip(), (path.name, buf.getvalue()[:200])
        assert callable(getattr(module, "main", None)), path.name


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, exc))
            print("FAIL  %s: %s" % (name, exc))
        except Exception as exc:                     # noqa: BLE001
            failed.append((name, exc))
            print("ERROR %s: %r" % (name, exc))
        else:
            print("ok    %s" % name)
    print()
    print("%d of %d checks passed" % (len(tests) - len(failed), len(tests)))
    if failed:
        return 1
    print("The installation looks good. `python run.py list` shows what to run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
