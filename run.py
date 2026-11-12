#!/usr/bin/env python3
"""Run the paper's experiments by the name of the result they produce.

The scripts under experiments/ keep the names they had when the results were
recorded, which are not meaningful to a reader of the paper. This dispatcher
puts a name on each one:

    python run.py list                     # what can be run, and what it produces
    python run.py table10                  # run it, writing out/table10.json
    python run.py table10 --out mine.json  # choose where it goes
    python run.py --all                    # every experiment that runs on one machine

Each experiment runs in its own process, because the scripts set their
configuration through environment variables at import time and two of them in
one process would interfere.

The five sweeps under experiments/compute_host/ are not listed here: they take
fifty seeds across eight configurations and have their own calibrate /
evaluate / aggregate steps. See the README.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Experiment:
    """One runnable experiment.

    `cost` states the configuration the script actually runs, in seeds and
    rounds, read from the script itself: a wall-clock estimate would depend on
    the machine. `args` carries the arguments needed to reproduce the
    configuration the paper reports, for the scripts whose own defaults differ
    from it.
    """

    def __init__(self, name: str, script: str, produces: str,
                 cost: str, args: tuple[str, ...] = ()) -> None:
        self.name = name
        self.script = script
        self.produces = produces
        self.cost = cost
        self.args = args

    @property
    def path(self) -> Path:
        return ROOT / self.script


EXPERIMENTS = [
    Experiment("table8", "experiments/mac5.py",
               "Table 8, marginal accuracy contribution per participant",
               "5 seeds x 3 rounds, 9 trainings per seed (leave-one-out)"),
    Experiment("table9", "experiments/p0_baseline.py",
               "Table 9, closed-loop gating and model recovery",
               "5 seeds x 3 rounds"),
    Experiment("table10", "experiments/p0_baseline.py",
               "Table 10, primary profile: conventional checks, norm filter, "
               "robust aggregation",
               "5 seeds x 3 rounds"),
    Experiment("table10-stealth", "experiments/p0_stealth2.py",
               "Table 10, stealth profile",
               "5 seeds x 3 rounds"),
    Experiment("table11", "experiments/e1_online.py",
               "Table 11, online adjudication over 25 rounds",
               "25 seeds x 25 rounds (seeds 77-101)",
               ("25", "77", "25")),
    Experiment("table11-pilot", "experiments/e1_online.py",
               "the five-seed pilot of the same experiment, which the paper "
               "keeps in the deposit but does not report",
               "5 seeds x 25 rounds (seeds 72-76)",
               ("25", "72", "5")),
    Experiment("table15", "experiments/exp12.py",
               "Table 15, detection by control family over four threat profiles",
               "5 seeds x 3 rounds x 4 profiles"),
    Experiment("table16", "experiments/e2_sdf03.py",
               "Table 16, four ways of supplying the label-distribution signal",
               "5 calibration + 20 evaluation seeds x 3 rounds"),
    Experiment("tableS5", "experiments/exp3.py",
               "Table S5, governance overhead at 3, 10, 25 and 50 rounds",
               "5 seeds x (3 + 10 + 25 + 50) rounds"),
    Experiment("tableS6", "experiments/exp45.py",
               "Table S6, level-weight sensitivity, and the seed-level intervals",
               "5 seeds x 3 rounds, re-scored under four weight vectors"),
    Experiment("tableS10", "experiments/persist.py",
               "Table S10, 25-round participation comparison",
               "5 seeds x 25 rounds"),
    Experiment("downweight", "experiments/downweight.py",
               "Section 6.1, quarantine against down-weighting",
               "5 seeds x 3 rounds x four weight multipliers"),
    Experiment("sdf03-dependence", "experiments/sdf03.py",
               "Section 6.3, how much of the detection rests on S-DF-03",
               "5 seeds x 3 rounds"),
    Experiment("sdf03-gradient", "experiments/sdf03est.py",
               "Section 6.3, the update-derived channel on the calibration seeds",
               "5 seeds x 3 rounds"),
    Experiment("sdf03-heldout", "experiments/sdf03_indep.py",
               "Section 6.3, the same channel with the threshold frozen, on "
               "held-out seeds",
               "5 calibration + 25 held-out seeds x 3 rounds"),
]

BY_NAME = {e.name: e for e in EXPERIMENTS}


def cmd_list() -> int:
    print("Experiments that run on a single machine:")
    print()
    for e in EXPERIMENTS:
        print("  %s" % e.name)
        print("      %s" % e.produces)
        print("      runs %s" % e.cost)
        print()
    print("Run one with:   python run.py <name>")
    print("Run them all:   python run.py --all")
    print()
    print("The fifty-seed sweeps behind Tables 12, 13, 19, 20 and 21 need a")
    print("compute host and their own calibrate / evaluate / aggregate steps;")
    print("the README has the commands.")
    print()
    print("To check the published numbers without running anything:")
    print("  python verify/reproduce_tables.py")
    return 0


def run_one(exp: Experiment, out: Path, passthrough: list[str]) -> int:
    if not exp.path.exists():
        print("run.py: %s is missing" % exp.script, file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(exp.path), str(out), *(passthrough or exp.args)]
    print("=" * 72)
    print("%s  ->  %s" % (exp.name, out))
    print("%s" % exp.produces)
    print("runs %s" % exp.cost)
    print("=" * 72, flush=True)
    started = time.time()
    code = subprocess.call(argv, cwd=str(ROOT))
    minutes = (time.time() - started) / 60.0
    if code == 0:
        print()
        print("%s finished in %.1f min; wrote %s" % (exp.name, minutes, out))
    else:
        print()
        print("%s exited with code %d after %.1f min"
              % (exp.name, code, minutes), file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", add_help=True,
        description="Run one of the paper's experiments by name.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python run.py list   shows every name and what it produces.")
    parser.add_argument("name", nargs="?", help="experiment name, or 'list'")
    parser.add_argument("--out", type=Path, default=None,
                        help="output file (default: out/<name>.json)")
    parser.add_argument("--all", action="store_true",
                        help="run every single-machine experiment in turn")
    args, passthrough = parser.parse_known_args(argv)

    if args.all:
        failed = []
        for exp in EXPERIMENTS:
            if exp.name == "table11-pilot":
                continue        # the paper does not report it
            out = ROOT / "out" / ("%s.json" % exp.name)
            if run_one(exp, out, []) != 0:
                failed.append(exp.name)
        print()
        if failed:
            print("failed: %s" % ", ".join(failed), file=sys.stderr)
            return 1
        print("every experiment finished; output in %s" % (ROOT / "out"))
        return 0

    if args.name in (None, "list", "help"):
        return cmd_list()

    exp = BY_NAME.get(args.name)
    if exp is None:
        print("run.py: no experiment called %r." % args.name, file=sys.stderr)
        print(file=sys.stderr)
        guesses = sorted(n for n in BY_NAME if args.name.lower() in n.lower())
        if guesses:
            print("did you mean: %s" % ", ".join(guesses), file=sys.stderr)
        print("run `python run.py list` to see the names.", file=sys.stderr)
        return 2

    out = args.out or (ROOT / "out" / ("%s.json" % exp.name))
    return run_one(exp, out, passthrough)


if __name__ == "__main__":
    sys.exit(main())
