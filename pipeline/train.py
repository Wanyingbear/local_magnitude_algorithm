"""Train global49/local49 RF and GBDT models, resuming validated checkpoints."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from .common import (CACHE, RESULTS, dataset, settings, splits, make_model,
                     model_spec, feature_signature, run_signature, checkpoint_path,
                     save_npz, save_json, environment, read_json, sha, DATA)
STATE = {}


def initialize():
    data = dataset()
    prep = read_json(CACHE/'preparation.json')
    if (prep['task2_data_sha256'] != sha(DATA/'task2.npz')
            or prep['dataset_metadata_sha256'] != sha(DATA/'datasets.json')
            or prep['settings'] != settings()):
        raise ValueError('Inputs or settings changed; run prepare first')
    STATE.clear()
    STATE['data'] = data
    STATE['x'] = {}
    for case in settings()['cases']:
        path = CACHE/'features'/f'{case}.npy'
        if sha(path) != prep['feature_hashes'][case]:
            raise ValueError('Feature cache changed; run prepare first')
        x = np.load(path, mmap_mode='r')
        if x.shape != (len(data['y']), 49) or not np.isfinite(x).all():
            raise ValueError('Invalid feature matrix')
        np.testing.assert_array_equal(x[:, 12:], data['auxiliary'].astype(np.float32))
        STATE['x'][case] = x
    STATE['signatures'] = {case:feature_signature(x, data) for case,x in STATE['x'].items()}


def valid_checkpoint(path, sig, test):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata'].item()))
            return (meta['signature'] == sig and np.array_equal(z['test_indices'], test)
                    and z['prediction'].shape == test.shape and np.isfinite(z['prediction']).all())
    except (OSError, ValueError, KeyError):
        return False


def fit(job):
    protocol, case, name, fold, sig = job
    data = STATE['data']
    for index, train, test in splits(data['groups'], protocol):
        if index == fold:
            break
    else:
        raise ValueError('Missing fold')
    if np.intersect1d(train, test).size:
        raise ValueError('Train/test overlap')
    if protocol != 'atom_cv' and np.intersect1d(data['groups'][train], data['groups'][test]).size:
        raise ValueError('Train/test protein overlap')
    with threadpool_limits(limits=1):
        model = make_model(name)
        model.fit(STATE['x'][case][train], data['y'][train])
        prediction = model.predict(STATE['x'][case][test])
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite prediction')
    meta = dict(signature=sig, protocol=protocol, case=case, model=name, fold=fold,
                origin='trained', parameters=model_spec(name), environment=environment())
    save_npz(checkpoint_path(protocol, case, name, fold), test_indices=test,
             prediction=prediction, training_mean=np.array(data['y'][train].mean()),
             metadata=np.array(json.dumps(meta)))
    return protocol, case, name, fold


def main(workers=4, protocols=None, aggregate_only=False, force_retrain=False):
    from .audit import summarize
    protocols = settings()['protocols'] if protocols is None else protocols
    if workers < 1 or not protocols or len(set(protocols)) != len(protocols):
        raise ValueError('Choose positive workers and distinct protocols')
    if force_retrain and aggregate_only:
        raise ValueError('Force retraining and aggregate-only are mutually exclusive')
    initialize()
    jobs, cached, imported = [], 0, 0
    for protocol in protocols:
        for fold, train, test in splits(STATE['data']['groups'], protocol):
            for case in settings()['cases']:
                for name in settings()['models']:
                    sig = run_signature(STATE['signatures'][case], protocol, name)
                    path = checkpoint_path(protocol, case, name, fold)
                    if not force_retrain and valid_checkpoint(path, sig, test):
                        cached += 1
                        with np.load(path) as z:
                            imported += json.loads(str(z['metadata'].item()))['origin'] == 'imported'
                    else:
                        jobs.append((protocol, case, name, fold, sig))
    print(f'New fits {len(jobs)}, cached {cached} (imported {imported}), workers {workers}', flush=True)
    if jobs and aggregate_only:
        raise RuntimeError('Missing or invalid checkpoints; aggregate-only cannot train them')
    if jobs:
        save_json(RESULTS/'validation.json', {'passed': False, 'status':'training; run audit after completion'})
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as pool:
            futures = [pool.submit(fit, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                last = future.result()
                if i % 10 == 0 or i == len(jobs):
                    print(f'{i}/{len(jobs)} fits completed; last={last}', flush=True)
    summarize(protocols, verify_task1=False)
    save_json(RESULTS/'run.json', dict(protocols=protocols, new_fits=len(jobs), cached=cached,
              imported_checkpoints_used=imported, force_retrain=force_retrain, environment=environment()))
    print('Results:', RESULTS, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--protocols', nargs='+', choices=['protein_cv','atom_cv','lopo'])
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--force-retrain', action='store_true')
    main(**vars(parser.parse_args()))
