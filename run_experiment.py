"""Run independent global/local preparation, training, or result validation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGE = 'audit'
WORKERS = 4
PROTOCOLS = ['protein_cv', 'atom_cv', 'lopo']
AGGREGATE_ONLY = False
FORCE_RETRAIN = False


def run_stage(stage, *, workers=4, protocols=None, aggregate_only=False, force_retrain=False, extra=None):
    modules = {'prepare': 'pipeline.prepare', 'train': 'pipeline.train', 'audit': 'pipeline.audit'}
    if stage not in modules:
        raise ValueError('Choose prepare, train, or audit')
    if extra is None:
        extra = []
        if stage in ('prepare', 'train'):
            extra += ['--workers', str(workers)]
        if stage in ('train', 'audit') and protocols is not None:
            extra += ['--protocols', *protocols]
        if stage == 'train':
            if aggregate_only:
                extra.append('--aggregate-only')
            if force_retrain:
                extra.append('--force-retrain')
    command = [sys.executable, '-u', '-m', modules[stage], *extra]
    print(f'Running {stage} with {sys.executable}', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    output = ROOT / 'results/k50'
    print(f'Finished {stage}. Results: {output}', flush=True)
    return output


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return run_stage(STAGE, workers=WORKERS, protocols=PROTOCOLS,
                         aggregate_only=AGGREGATE_ONLY, force_retrain=FORCE_RETRAIN)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'train', 'audit'])
    args, extra = parser.parse_known_args(argv)
    return run_stage(args.stage, extra=extra)


if __name__ == '__main__':
    results_directory = main()
