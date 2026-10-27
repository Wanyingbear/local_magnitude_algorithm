"""Press F5 to verify packaged inputs and independent weighting solves."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithms import local_ca_regular, global_ca
from pipeline.common import sha
CHECK_FILE_HASHES = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-hashes', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Tests always run')
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv) if argv else argparse.Namespace(skip_hashes=not CHECK_FILE_HASHES)
    manifest = json.loads((ROOT/'manifest.json').read_text(encoding='utf8'))
    if not args.skip_hashes:
        failures = [r['path'] for r in manifest['files'] if not (ROOT/r['path']).is_file() or sha(ROOT/r['path']) != r['sha256']]
        if failures:
            raise ValueError('Missing or modified files: ' + ', '.join(failures))
    xyz = np.random.default_rng(7).normal(size=(10, 3))*4
    alphas = (.1, .2, .3, .4, .5, .6)
    distance = cdist(xyz, xyz)
    expected = np.column_stack([solve(np.exp(-a*distance**p), np.ones(len(xyz)), assume_a='gen') for p in (1, 2) for a in alphas])
    glob = global_ca(xyz, alphas=alphas)
    loc = local_ca_regular(xyz, k=50, alphas=alphas)
    np.testing.assert_allclose(glob, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(loc, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(local_ca_regular(xyz, k=1), np.ones((10, 12)), atol=1e-12)
    result = dict(passed=True, hashes_checked=not args.skip_hashes, full_neighborhood_matches_global=True,
                  max_lu_error=float(np.max(abs(loc-expected))))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    validation = main()
