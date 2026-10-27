[中文说明](README_ZH.md)

# Local Magnitude

Standalone global/local C-alpha weighting and B-factor experiments. No external repository or baseline script is required. The package keeps runtime code, fixed inputs, cache and current results only; the manuscript and provenance notes are maintained outside it.

## Run

In Spyder press F5 in spyder_prepare.py, spyder_train.py, then spyder_audit.py. Existing features permit train then audit. Set WORKERS=4 (or 2), PROTOCOLS, AGGREGATE_ONLY and FORCE_RETRAIN in spyder_train.py. Force retraining fits BOTH cases for the selected protocols and replaces checkpoints. Otherwise valid package checkpoints resume; missing predictions cause aggregate-only to fail. No external baseline is reused.

The full experiment has 1460 models: two representations, two models, 10 protein folds, 10 atom folds and 345 LOPO folds. RF/GBDT use 1000 trees and seed 0. Packaging did not retrain all models. Imported origins remain explicit.

## Outputs

Read results/k50/: task1_summary.csv, task1_metrics.csv, task2_metrics.csv, comparison.csv, task2_per_protein.csv, lopo_subsets.csv, feature_diagnostics.csv, validation.json, run.json and REPORT.md. Only k50 is evaluated. No search results, duplicate per-atom CSVs or dataset-wide timings are generated.

cache/k50 contains feature arrays, Task1 fits and compressed prediction checkpoints. Each NPZ includes test_indices, prediction, training_mean and metadata. Join indices to data/task2.npz for identities/targets. Complete validation reports passed=true, complete=true and 1460 folds; partial audits report scope and complete=false. Preparing or retraining invalidates prior verification first.

## Definitions

algorithms/ provides local_ca_regular and global_ca. Inputs: distinct finite (N,3) C-alpha coordinates in angstroms. K=exp(-alpha\*distance\*\*p), p=1,2; Cholesky solves, no inverse or silent jitter. Local returns center weights, not neighborhood totals.

Local: k50, alpha=0.1,...,0.6. Global: alpha=0.3,...,0.8, preserving the manuscript configuration. Differences combine locality and scale. Task1: 364 proteins/78403 rows, 12-column within-protein fits. Task2: 345 proteins/74527 rows, 12 geometry plus 37 identical supplied auxiliary columns. Targets are separate from inputs. Auxiliary generation for arbitrary new structures is not included.

Geometry uses float64; tree inputs are stored in float32. Ten-fold primary_pcc averages folds; LOPO averages proteins. Pooled PCC differs. Atom CV can train on other residues of the test protein. Single-seed evaluation and historical selection on overlapping proteins are exploratory, not independent external validation. Library versions can affect refits.

pipeline/ contains preparation, training and auditing. data/task1 contains xyz,y,source_rows; data/task2.npz contains xyz,y,groups,auxiliary,source_rows; data/datasets.json defines cohorts. experiment.json holds settings.

results/k50 holds current output. The package ships no LaTeX manuscript or fixed evidence snapshot, so paper numbers are maintained in the external manuscript directory and are never rewritten here by a run.

## Utilities

Use `local_ca_regular(xyz, k=50, n_jobs=4)` for neighborhood threading; the default remains `n_jobs=1`. Wrap threaded calls in `threadpoolctl.threadpool_limits(limits=1)` to avoid nested BLAS parallelism.

verify_bundle.py checks fixed file hashes and LU/full-neighborhood identities. The package holds only the Task1/Task2 chain: no timing scripts, example datasets or bytecode caches.

```text
python -m pip install numpy pandas scipy scikit-learn threadpoolctl
python verify_bundle.py
python run_experiment.py prepare --workers 4
python run_experiment.py train --workers 4
python run_experiment.py train --protocols protein_cv --force-retrain --workers 2
python run_experiment.py audit
```

The manifest covers algorithms and fixed inputs, not mutable results. After intentional changes use verify_bundle.py --skip-hashes.
