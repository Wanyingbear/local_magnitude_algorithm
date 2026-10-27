"""Edit settings and press F5 to run Task2 training using valid caches."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_experiment import run_stage

# Select protocols and process count before pressing F5.
WORKERS = 4
PROTOCOLS = ['protein_cv', 'atom_cv', 'lopo']
AGGREGATE_ONLY = False  # True only summarizes valid existing checkpoints.
FORCE_RETRAIN = False  # True retrains both cases and replaces selected checkpoints.

if __name__ == '__main__':
    results_directory = run_stage('train', workers=WORKERS, protocols=PROTOCOLS,
                                  aggregate_only=AGGREGATE_ONLY, force_retrain=FORCE_RETRAIN)
