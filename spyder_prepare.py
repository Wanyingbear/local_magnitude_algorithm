"""Press F5 to recompute Task1 and prepare Task2 features (may take time)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_experiment import run_stage

if __name__ == '__main__':
    results_directory = run_stage('prepare')
