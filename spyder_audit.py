"""Press F5 to independently audit predictions and regenerate the report."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from run_experiment import run_stage

if __name__ == '__main__':
    results_directory = run_stage('audit')
    validation = json.loads((results_directory / 'validation.json').read_text(encoding='utf8'))
