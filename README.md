# Local Magnitude

[中文说明](README_ZH.md)

Standalone global/local C-alpha weighting and B-factor experiment package. The whole directory is self-contained and movable; no external repository, original scripts or previous workspace outputs are required. The package keeps runtime code, fixed inputs, cache and current results only; the manuscript and provenance notes are maintained outside it.

## Run order in Spyder

1. spyder_prepare.py — compute both features from coordinates, finish Task1 and prepare Task2.
2. spyder_train.py — train RF/GBDT for global49/local49.
3. spyder_audit.py — independently check predictions, splits and scoring.

Open a file and press F5. With existing features you can go directly train → audit. The interpreter needs numpy, pandas, scipy, scikit-learn and threadpoolctl.

Settings in spyder_train.py: WORKERS=4 (2 is fine on a typical laptop), PROTOCOLS=['protein_cv','atom_cv','lopo'], AGGREGATE_ONLY=False, FORCE_RETRAIN=False.

- FORCE_RETRAIN=False: reuse valid in-package caches; prints New fits, cached, imported.
- FORCE_RETRAIN=True: refit both global and local for the selected protocols and replace checkpoints; no external baseline reuse. Back up cache/k50 first if you want to keep old runs.
- AGGREGATE_ONLY=True: aggregate complete valid caches only; fails if anything is missing.
- The full experiment has 1460 models, RF/GBDT with 1000 trees each, seed=0. To try ten-fold first, select only protein_cv and atom_cv. Imported cache is not the same as new training.

## Outputs

**Just look at results/k50/**:

| File | Purpose |
| --- | --- |
| task1_summary.csv | Task1 scores on the four subsets |
| task1_metrics.csv | Task1 per-protein results and scatter data |
| task2_metrics.csv | Primary PCC, pooled PCC, RMSE, MAE and mean baseline |
| comparison.csv | Six direct global/local comparisons, no literature values mixed in |
| task2_per_protein.csv | Task2 per-protein scores |
| lopo_subsets.csv | LOPO grouping |
| feature_diagnostics.csv | Feature ranges and saturation |
| validation.json | Validation status of this run |
| run.json | New fits and cached counts of the last training |
| REPORT.md | Concise result summary |

A complete audit reports passed=true, complete=true, fold_prediction_files_checked=1460, with every test row covered exactly once. Partial-protocol audits mark scope and complete=false. prepare/train invalidate prior verification before changing outputs.

Only the k50 experiment is kept. No radius-12Å style parameter searches, duplicate per-atom CSVs, staged reports or dataset-wide timings are generated. The 1460 compressed prediction files used for resuming and verification live in cache/k50/task2; each contains test_indices, prediction, training_mean and metadata. Join test_indices to data/task2.npz to recover atom identities and targets.

cache/k50 contains feature arrays, Task1 fits and compressed prediction checkpoints. Complete validation reports passed=true, complete=true and 1460 folds; partial audits report scope and complete=false. Preparing or retraining invalidates prior verification first.

## Configuration and definitions

- local: 50 nearest C-alpha including the center, p=1,2, alpha=0.1,...,0.6; use all C-alpha when fewer than 50.
- global: p=1,2, alpha=0.3,...,0.8, preserving the manuscript configuration.
- K=exp(-alpha\*d\*\*p), Cholesky solves Kw=1; local returns center weights, not neighborhood totals.
- Task1: 364 proteins / 78403 rows, 12-column within-protein OLS fits on own labels, not blind testing.
- Task2: 345 proteins / 74527 rows, 12 geometry columns plus the same 37 supplied auxiliary columns; targets are never features.
- Geometry uses float64; tree inputs are cached as float32, matching the model's internal representation.
- Ten-fold primary_pcc averages folds; LOPO averages proteins; pooled PCC is not interchangeable.
- Atom CV may train on other residues of the test protein. Single-seed evaluation and historical selection on overlapping proteins are exploratory, not independent external validation.
- Library versions can affect refits; imported caches keep explicit origins and audit never retrains.

## Directory

```text
algorithms/          local_ca_regular.py, global_ca.py
pipeline/            prepare, train, audit, common
data/task1/          xyz, y, source_rows
data/task2.npz       xyz, y, groups, auxiliary, source_rows
data/datasets.json   cohort, subset and auxiliary column definitions
cache/k50/           features, fits and prediction checkpoints
results/k50/         current run output
experiment.json      unified configuration
```

The package keeps only the Task1/Task2 chain: runtime code, fixed inputs, cache and results; timing scripts, example datasets and bytecode caches are excluded.

results/k50 holds current output. The package ships no LaTeX manuscript or fixed evidence snapshot, so paper numbers are maintained in the external manuscript directory and are never rewritten here by a run.

## Direct algorithm calls and verification

`local_ca_regular(xyz, k=50, n_jobs=4)` supports neighborhood threading; the default remains `n_jobs=1`. Wrap threaded calls in `threadpoolctl.threadpool_limits(limits=1)` to avoid nested BLAS parallelism.

verify_bundle.py checks fixed-input/algorithm hashes and independent LU identities. After intentional source edits use --skip-hashes. The manifest does not lock mutable results.

```python
from algorithms import local_ca_regular, global_ca
local = local_ca_regular(xyz, k=50)
global_weights = global_ca(xyz)
```

```text
python -m pip install numpy pandas scipy scikit-learn threadpoolctl
python verify_bundle.py
python run_experiment.py prepare --workers 4
python run_experiment.py train --workers 4
python run_experiment.py train --protocols protein_cv --force-retrain --workers 2
python run_experiment.py audit
```
