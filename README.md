# A DLT-Based Governance Framework for Agent Trustworthiness

Reference implementation and experiment drivers for the paper

> **Hierarchical Trustworthiness Evaluation for Distributed Multi-Agent Systems:
> A DLT-Based Governance Framework Evaluated in Federated Learning**
> Soonduck Yoo and Dae-Yeol Kim

Everything reported in the paper can be re-run from this repository. Each table
is produced by a named script, listed in [Reproducing the paper](#reproducing-the-paper),
and the run outputs behind the published numbers are committed under
[`results/`](results) so you can check a figure without re-running anything.

---

## What the framework does

The framework separates two questions that trust-management methods usually
answer together: *is this contribution useful?* and *was this participant
entitled to make it?*

- **A 3-12-36 control specification.** Agent trustworthiness is decomposed into
  three levels — `Structural` (the system's governance foundation), `Behavioral`
  (whether an agent acted within its authorization) and `Granular` (whether an
  individual submission is technically valid) — operationalized as 12 categories
  of 3 control points each, 36 in total. Every control is tied to a standards
  clause and marked as **measured** at runtime or resting on operator
  **attestation**.
- **Severity-based, non-compensatory adjudication.** A failed control carries
  **Reject**, **Review** or **Info** severity. One Reject failure settles the
  round regardless of scores elsewhere, so a critical violation cannot be
  offset by a high score on an unrelated control. This ordering is the
  established zero-trust policy-decision pattern (NIST SP 800-207) applied to
  an enumerated control surface; it is not claimed as new.
- **Verdict kept apart from reputation.** The discrete per-round verdict
  (`Trusted` / `Under Review` / `Untrusted`) is recorded separately from the
  continuous reputation score, which is used for ranking, monitoring and
  recovery but never decides the current round.
- **An append-only evidence record.** Each decision stores the violated
  requirement, the per-control evidence, the severity, the action applied, the
  effect on reputation, and enough of the ledger to rebuild the decision later.

### What the evaluation does and does not establish

The paper reports the results that do not favour the framework alongside those
that do, and this repository reproduces both.

- Conventional security checks (authentication tag, payload hash, freshness,
  credential revocation and expiry) **with one peer-relative norm filter already
  identify four of the five injected violators** at 0.8392 accuracy, against
  0.8524 and five of five for the full control set. The paired accuracy
  difference over the same five seeds is +0.0132 ± 0.0084 with an exact
  two-sided *p* = 0.125, which five seeds cannot establish as an advantage.
- **Robust aggregation alone is the most accurate configuration tested**
  (Trimmed-Mean 0.8686, Multi-Krum 0.8644) while issuing no participant-level
  decision at all.
- **Only 7 of the 36 control points fire** under the injected threats; disabling
  the other 29 changes no verdict in any of 2,800 runs. The remaining 29 are
  claimed as coverage of standards clauses, not as measured detection benefit.
- The one control that distinguishes the framework from a conventional pipeline,
  S-DF-03, **reads a label distribution a deployed server does not hold.** A
  server-side substitute recovered from the submitted update does carry it over
  25 to 50 rounds, but **flags more honest participants** while doing so, and
  **under heterogeneous data no label-distribution channel tested is better than
  running none at all.**
- **Permanent exclusion is not a defensible default:** it triples the
  participant-rounds in which an honest participant is wrongly excluded.
- **Auditability is not a differentiator.** A conventional pipeline that retains
  the same evidence reproduces the same verdicts, so what differs is which
  evidence is retained rather than the ledger.
- The testbed implements an append-only hash chain with SHA-256 payload hashes
  and HMAC-SHA256 authentication tags **in a single process. It runs no
  consensus protocol**, so reported latencies exclude network broadcast and
  finality, and the throughput, fault tolerance and cost of a deployed DLT are
  not evaluated. The replicated-ledger condition measures the value of
  independent custody of several copies, not the behaviour of any consensus
  protocol.

---

## Quick start

You need **Python 3.12**. `numpy==1.26.4` has no wheels for Python 3.13 or
later, so a 3.12 interpreter is required to reproduce the reported numbers.

```sh
git clone https://github.com/SpicyYeol/A-DLT-Based-Governance-Framework.git
cd A-DLT-Based-Governance-Framework

sh scripts/setup_env.sh                 # or: sh scripts/setup_env.sh /path/to/python3.12
. .venv/bin/activate                    # Windows: .venv\Scripts\activate

python verify/reproduce_tables.py        # re-derive published values, no experiments run
python experiments/p0_baseline.py out.json   # re-run the layered baseline comparison
```

On Windows without a POSIX shell, the setup is three commands:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
mkdir data\exp, data\exp12
```

### Datasets

MNIST (OpenML id 554) and Fashion-MNIST (OpenML id 40996) are fetched by
identifier on first use through `sklearn.datasets.fetch_openml` and cached under
`$TRUST_DATA_ROOT/dataset_cache`. No dataset is redistributed here, and `data/`
is git-ignored. The first run needs network access; later runs are offline.

### Configuration

Every driver sets what it needs, but two variables are worth knowing about.

| Variable | Default | Notes |
|---|---|---|
| `TRUST_DATA_ROOT` | `<repo>/data/exp` (set by each driver) | dataset cache, ledger, reports |
| `TRUST_IMAGE_CAP` | **8000** in `src/trust/config.py` | the drivers set **4000**, which is the paper's budget |
| `TRUST_TESTBED` | the repository root | override only if you move `src/` or `data/` |
| `TRUST_ANOMALY_Z_TAU` | `3.0` | the anomaly threshold τ used throughout |

The image budget matters: the default in `config.py` is 8,000, every reported
table uses **4,000**, and the drivers set it explicitly. The paper documents a
silent fallback to that 8,000 default as the cause of an inconsistency it had to
trace, so do not rely on the default when comparing against the article.

---

## Reproducing the paper

Run from the repository root with the virtual environment active. Scripts that
take `<out.json>` write one JSON file; the committed output of the same run is
named in the last column.

### Main text

| Paper table | Script | Committed output |
|---|---|---|
| Table 8 — per-agent verdicts and marginal accuracy contribution | `experiments/mac5.py <out.json>` | `results/local/mac5_out.json` |
| Table 9 — closed-loop gating and model recovery | `experiments/p0_baseline.py <out.json>`, `experiments/exp45.py <out.json>` | `results/local/p0_out.json`, `exp45_out.json` |
| Table 10 — conventional checks, norm filter and robust aggregation | `experiments/p0_baseline.py` (primary), `experiments/p0_stealth2.py` (stealth) | `results/local/p0_out.json`, `p0_stealth2_out.json` |
| Table 11 — online adjudication over 25 rounds | `experiments/e1_online.py <out.json>` | `results/local/e1_indep25.json` (held-out), `e1_valid5.json` |
| Table 12 — deployable evidence channels, 50 seeds | `experiments/compute_host/e4_deployable.py` | `results/compute_host/e4_summary.json` |
| Table 13 — enforcement policies A–E | `experiments/compute_host/e5_policy.py` | `results/compute_host/e5_summary.json` |
| Table 14 — control and layer ablation | `scripts/run_experiments.py` (`run_control_ablation_full`) | CSV under `data/reports/` |
| Table 15 — detection by control family | `experiments/exp12.py <out.json>` | `results/local/exp12_out.json` |
| Table 16 — four label-distribution channels | `experiments/e2_sdf03.py <out.json>` | `results/local/e2_full.json` |
| Table 17 — non-IID Dirichlet sweep | `scripts/run_experiments.py` (`run_noniid`, `run_noniid_fix`) | CSV under `data/reports/` |
| Table 18 — overhead and large-scale extension | `scripts/run_experiments.py` (`run_scalability`), `scripts/rev1_scale.py` | CSV under `data/reports/` |
| Table 19 — participant count, local data held constant | `experiments/compute_host/e7_scale.py` | `results/compute_host/e7_summary.json` |
| Table 20 — ledger ablation under post-hoc manipulation | `experiments/compute_host/e6_audit.py`, `scripts/rev1_ledger.py` | `results/compute_host/e6_summary.json` |
| Table 21 — auditability, four records under a policy change | `experiments/compute_host/e6b_audit.py` | `results/compute_host/e6b_summary.json` |

### Supplementary tables and in-text results

| Where | Script | Committed output |
|---|---|---|
| Table S2 — the 30% manipulation configurations | `experiments/compute_host/e4_deployable.py` (same sweep) | `results/compute_host/e4_summary.json` |
| Table S3 — proportion of problematic participants | `scripts/run_experiments.py` (`run_fraction_sweep`) | CSV under `data/reports/` |
| Table S4 — backbones and datasets | `scripts/run_experiments.py` (`run_cross_dataset`, `run_deep_backbone`) | CSV under `data/reports/` |
| Table S5 — overhead against round count | `experiments/exp3.py <out.json>` | `results/local/exp3_out.json` |
| Table S6 — level-weight sensitivity | `experiments/exp45.py <out.json>` | `results/local/exp45_out.json` |
| Table S10 — 25-round participation comparison | `experiments/persist.py <out.json>` | `results/local/persist_out.json` |
| §6.1 quarantine against down-weighting | `experiments/downweight.py <out.json>` | `results/local/downweight_out.json` |
| §6.3 dependence on S-DF-03 | `experiments/sdf03.py <out.json>` | `results/local/sdf03_out.json` |
| §6.3 gradient channel, calibration seeds | `experiments/sdf03est.py <out.json>` | `results/local/sdf03est_out.json` |
| §6.3 gradient channel, frozen threshold on held-out seeds | `experiments/sdf03_indep.py <out.json>` | `results/local/sdf03_indep_out.json` |
| §6.4 collusion breakdown | `scripts/rev1_collusion.py` | CSV under `data/reports/` |
| §6.5 attestation band re-scoring, combined baseline | `scripts/run_extra.py` | printed |
| §6.1 seed-level intervals | `experiments/exp45.py`, `scripts/rev1_seeds.py` | `results/local/exp45_out.json` |

The four `scripts/rev1_*.py` runners implement the collusion check, the
extended 30-seed evaluation, the participant sweep to 1,024 and the ledger
ablation. They were written during an earlier revision round for a different
venue, so their console headers name that venue; the experiments they run are
the ones this paper reports.

### Experiments that need a compute host

The five drivers under `experiments/compute_host/` sweep fifty seeds across
eight configurations and were run on a multi-core host (environment B). They
take a subcommand rather than a single output path, so each cell and seed can be
run as its own task and the results aggregated afterwards:

```sh
# Table 12: calibrate, freeze the thresholds, evaluate each cell, aggregate
python experiments/compute_host/e4_deployable.py calib  <cell> <seed> <out.jsonl>
python experiments/compute_host/e4_deployable.py freeze <calib_dir> <thresholds.json>
python experiments/compute_host/e4_deployable.py eval   <cell> <cond> <seed> <thresholds.json> <out.jsonl>
python experiments/compute_host/e4_deployable.py agg    <eval_dir> <thresholds.json> <out.json>

# Tables 13, 19, 20 and 21 follow the same run / agg shape
python experiments/compute_host/e5_policy.py run <cell> <ctrlset> <policy> <seed> <out.jsonl>
python experiments/compute_host/e5_policy.py agg <eval_dir> <out.json>
python experiments/compute_host/e7_scale.py  run <n_agents> <cond> <seed> <out.jsonl>
python experiments/compute_host/e6b_audit.py run <cell> <seed> <thresholds.json> <out.jsonl>
```

Pass `--help`-less scripts no arguments to print their own usage block.

`python experiments/compute_host/verify_env.py` prints the interpreter and
library versions, which is worth running first on a new host.

Per-seed records for these runs are committed as
`results/compute_host/e4e7_perseed.tar.gz` (5,100 records), so the aggregation
step can be checked without re-running the sweeps.

---

## Reproducibility contract

**Two environments, never pooled.** Environment A is Python 3.12.3 with the four
pinned library versions in `requirements.txt`; environment B is the same four
versions on Python 3.12.14 and a different processor. Re-running the online
adjudication on B reproduced every detection, isolation, wrongly-excluded and
re-admission count, and moved accuracies by between −0.012 and +0.006. Because
the two are not bit-identical, **accuracies from them are never pooled**, and
every comparison in the paper is made within one environment on the same seeds.

**Seed discipline.** Three rules hold throughout:

1. Paired protocols share seeds deliberately — 42–46 for the five-seed
   protocols, 42–71 for the extended protocol — because a shared seed is what
   makes those comparisons paired.
2. A calibration set never shares a seed with the evaluation set that tests the
   threshold fitted on it.
3. Every independent evaluation draws seeds from a range no other experiment
   uses: 77–101 (online adjudication), 102–106 / 110–129 (channels),
   150–159 / 200–249 (deployable channels), 300–349 (policies), 500–519
   (participant count), 600–619 (auditability).

Thresholds reported as frozen were written to disk before the corresponding
evaluation seeds were run.

**Standard deviations** use the population estimator over the evaluated seeds,
dividing by *n* rather than *n* − 1, in every table and in the text.

**Runs kept but not reported.** Seeds 72–76 are a five-seed pilot of the online
experiment. Seeds 400–419 hold a two-record form of the auditability experiment,
a design that cannot separate the breadth of the retained evidence from its
privilege, which is why the reported experiment uses four records on seeds
600–619. Both are kept so the deposit accounts for every run; neither is
reported as a result.

---

## Repository layout

```
src/trust/              the framework
  taxonomy.py           the 36 control points, their levels, categories and severities
  controls/             per-level control implementations (small / medium / large)
  engine.py             control evaluation, scoring and verdict adjudication
  checks_base.py        shared predicate helpers
  ledger.py             append-only hash chain, SHA-256 hashes, HMAC-SHA256 tags
  artifacts.py          intermediate signals passed between stages
  attestations.py       operator-attested control evidence
  fl_simulation.py      the federated-learning testbed
  aggregators.py        FedAvg, Krum, Multi-Krum, Trimmed-Mean, Coordinate-Median
  fltrust.py            FLTrust, used as a comparison only
  datasets.py           OpenML fetch and cache
  experiments.py        the named experiment runners
  ml_eval.py            accuracy, macro-F1, worst-class recall, calibration error
  manifold.py           post-training distribution checks
  report.py             CSV and JSON writers
  tabular_security.py   the tabular detection path
  config.py             all configuration, read from the environment

experiments/            one driver per reported experiment
experiments/compute_host/   the fifty-seed sweeps (run / agg subcommands)
scripts/                environment setup and the testbed's own experiment runners
audits/                 scripts that audited two specific claims (see below)
results/                the run outputs behind the published numbers
verify/                 re-derive published values from those outputs
```

### `audits/`

Two scripts answer questions about the framework's own reporting rather than
producing a table, and are kept because the paper refers to what they found:

- `weightcheck.py` — audits the claim that per-agent aggregation weight is
  capped at 0.30. The aggregator applies no cap; the 0.30 belongs to the
  influence control M-SC-02, where it is a detection threshold. No experimental
  result depends on this, since the cap was never implemented.
- `capid.py` — identifies which image budget reproduces a given set of primary
  figures, which is how the budget inconsistency described above was traced.

---

## Citation

See [`CITATION.cff`](CITATION.cff). The paper is under review, so no DOI is
given yet.

## License

[MIT](LICENSE). The datasets are not redistributed here and remain under their
own terms.
