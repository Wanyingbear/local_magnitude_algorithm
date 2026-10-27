"""Recompute global and local features directly from bundled coordinates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from algorithms import local_ca_regular, global_ca
from .common import (ROOT, DATA, CACHE, RESULTS, read_json, save_json, save_npz,
                     sha, signature, settings, dataset, fit_linear, environment)


def independent_check(xyz, local, config):
    error = 0.0
    for center in sorted({0, len(xyz)//2, len(xyz)-1}):
        distances = np.linalg.norm(xyz - xyz[center], axis=1)
        others = np.delete(np.arange(len(xyz)), center)
        ids = np.r_[center, others[np.lexsort((others, distances[others]))]][:config['k']]
        distance = cdist(xyz[ids], xyz[ids])
        for pi, power in enumerate((1, 2)):
            for j, alpha in enumerate(config['local_alphas']):
                kernel = np.exp(-alpha * distance**power)
                w = solve(kernel, np.ones(len(ids)), assume_a='gen')
                error = max(error, float(abs(w[0]-local[center, 6*pi+j])))
    if error > 1e-9:
        raise ValueError('Independent LU validation failed')
    return error


def calculate(job):
    task, protein = job
    config = settings()
    if task == 'task1':
        with np.load(DATA / 'task1' / f'{protein}.npz') as z:
            xyz, y = z['xyz'], z['y']
    else:
        data = dataset()
        select = data['groups'] == protein
        xyz, y = data['xyz'][select], data['y'][select]
    with threadpool_limits(limits=1):
        local = local_ca_regular(xyz, k=config['k'], alphas=config['local_alphas'])
        glob = global_ca(xyz, alphas=config['global_alphas'])
        lu_error = independent_check(xyz, local, config)
        if task == 'task1':
            lp, ls = fit_linear(local, y)
            gp, gs = fit_linear(glob, y)
            save_npz(CACHE / 'task1' / f'{protein}.npz', local_features=local,
                     global_features=glob, local_prediction=lp, global_prediction=gp)
            return dict(protein=protein, lu_error=lu_error, rows=[
                dict(protein=protein, method='global12', **gs),
                dict(protein=protein, method='local12', **ls)])
    return dict(protein=protein, local=local, global_features=glob, lu_error=lu_error)


def main(workers=4):
    if workers < 1:
        raise ValueError('workers must be positive')
    config = settings()
    meta = read_json(DATA / 'datasets.json')
    data = dataset()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (CACHE / 'features').mkdir(parents=True, exist_ok=True)
    # Invalidate previous verification before replacing any dependent data.
    save_json(RESULTS / 'validation.json', {'passed': False, 'status': 'preparing; run audit after training'})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows, errors = [], []
        jobs = [('task1', p) for p in meta['task1_proteins']]
        for i, result in enumerate(pool.map(calculate, jobs), 1):
            rows.extend(result['rows']); errors.append(result['lu_error'])
            if i % 50 == 0 or i == len(jobs):
                print(f'Task1: {i}/{len(jobs)} proteins', flush=True)
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS / 'task1_metrics.csv', index=False)
        summary = []
        for subset, ids in meta['task1_subsets'].items():
            for method, g in table[table.protein.isin(ids)].groupby('method'):
                summary.append(dict(subset=subset, method=method, proteins=len(g),
                    mean_pcc=float(g.pcc.mean()), rmse=float(np.sqrt(np.average(g.rmse**2, weights=g.n))),
                    mae=float(np.average(g.mae, weights=g.n))))
        pd.DataFrame(summary).to_csv(RESULTS / 'task1_summary.csv', index=False)
        proteins = pd.unique(data['groups'])
        local = np.empty((len(data['y']), 12)); glob = np.empty_like(local)
        for i, result in enumerate(pool.map(calculate, [('task2', p) for p in proteins]), 1):
            ids = np.flatnonzero(data['groups'] == result['protein'])
            local[ids] = result['local']; glob[ids] = result['global_features']
            errors.append(result['lu_error'])
            if i % 50 == 0 or i == len(proteins):
                print(f'Task2: {i}/{len(proteins)} proteins', flush=True)
    # Both regressors use float32 internally; keeping that representation saves space.
    for name, x in [('global49', glob), ('local49', local)]:
        np.save(CACHE / 'features' / f'{name}.npy', np.c_[x, data['auxiliary']].astype(np.float32))
    diagnostics = [dict(power=power, alpha=alpha, minimum=float(local[:,i].min()),
                       maximum=float(local[:,i].max()), std=float(local[:,i].std()))
                   for i, (power, alpha) in enumerate((p,a) for p in (1,2) for a in config['local_alphas'])]
    pd.DataFrame(diagnostics).to_csv(RESULTS / 'feature_diagnostics.csv', index=False)
    save_json(CACHE / 'preparation.json', dict(status='recomputed', settings=config,
              task2_data_sha256=sha(DATA/'task2.npz'), dataset_metadata_sha256=sha(DATA/'datasets.json'),
              task1_data_sha256={p:sha(DATA/'task1'/f'{p}.npz') for p in meta['task1_proteins']},
              feature_hashes={c:sha(CACHE/'features'/f'{c}.npy') for c in config['cases']},
              max_lu_error=max(errors), environment=environment()))
    print('Prepared Task1 fits and Task2 global/local features. Run train, then audit.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    main(**vars(parser.parse_args()))
