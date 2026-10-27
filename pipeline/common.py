"""Shared dataset schema, signatures, metrics, and regression settings."""
from pathlib import Path
import hashlib
import json
import os
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import lstsq
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CACHE = ROOT / 'cache/k50'
RESULTS = ROOT / 'results/k50'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def save_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    array = np.ascontiguousarray(value)
    header = json.dumps([array.shape, array.dtype.str]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def settings():
    config = read_json(ROOT / 'experiment.json')
    if config['k'] != 50 or config['powers'] != [1, 2]:
        raise ValueError('This experiment package is restricted to k50 and p=1,2')
    return config


def dataset():
    with np.load(DATA / 'task2.npz', allow_pickle=False) as z:
        value = {key: z[key] for key in z.files}
    n = len(value['y'])
    if value['xyz'].shape != (n, 3) or value['auxiliary'].shape != (n, 37):
        raise ValueError('Dataset coordinate or auxiliary shape mismatch')
    if len(value['groups']) != n or len(value['source_rows']) != n:
        raise ValueError('Dataset identity shape mismatch')
    for key in ['xyz', 'auxiliary', 'y']:
        if not np.isfinite(value[key]).all():
            raise ValueError('Nonfinite dataset values: ' + key)
    return value


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__)


def mean_valid(values):
    finite = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(finite)) if finite else None


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid prediction shape or values')
    pcc = None if len(y) < 2 or np.ptp(y) < 1e-12 or np.ptp(prediction) < 1e-12 else float(np.corrcoef(y, prediction)[0, 1])
    return dict(n=len(y), pcc=pcc, rmse=float(np.sqrt(np.mean((y-prediction)**2))),
                mae=float(np.mean(np.abs(y-prediction))))


def fit_linear(x, y):
    design = np.asarray(x, dtype=float) - np.mean(x, axis=0)
    norms = np.linalg.norm(design, axis=0)
    valid = norms > 1e-12
    design[:, valid] /= norms[valid]
    design[:, ~valid] = 0
    coef, _, rank, _ = lstsq(design, y-y.mean(), cond=1e-10)
    prediction = design @ coef + y.mean()
    return prediction, dict(**metrics(y, prediction), rank=int(rank))


def model_spec(name):
    config = settings()
    base = dict(n_estimators=config['trees'], random_state=config['seed'])
    if name == 'RF':
        return dict(**base, max_depth=8, min_samples_split=4, min_samples_leaf=2,
                    max_features=1.0, n_jobs=1)
    if name == 'GBDT':
        return dict(**base, max_depth=7, min_samples_split=5, min_samples_leaf=1,
                    learning_rate=.002, subsample=.8, max_features='sqrt')
    raise ValueError(name)


def make_model(name):
    cls = RandomForestRegressor if name == 'RF' else GradientBoostingRegressor
    return cls(**model_spec(name))


def splits(groups, protocol):
    proteins = pd.unique(groups)
    if protocol == 'lopo':
        for fold, protein in enumerate(proteins):
            yield fold, np.flatnonzero(groups != protein), np.flatnonzero(groups == protein)
        return
    if protocol not in ('protein_cv', 'atom_cv'):
        raise ValueError(protocol)
    base = proteins if protocol == 'protein_cv' else groups
    for fold, (train, test) in enumerate(KFold(10, shuffle=True, random_state=settings()['seed']).split(base)):
        if protocol == 'protein_cv':
            train = np.flatnonzero(np.isin(groups, proteins[train]))
            test = np.flatnonzero(np.isin(groups, proteins[test]))
        yield fold, train, test


def feature_signature(x, data):
    return dict(features_float32=array_hash(np.asarray(x, dtype=np.float32)),
                target=array_hash(data['y']), groups=array_hash(data['groups']),
                rows=array_hash(data['source_rows']))


def run_signature(inputs, protocol, model):
    return signature(dict(schema=1, inputs=inputs, protocol=protocol,
                          split_seed=settings()['seed'], model=model, parameters=model_spec(model)))


def checkpoint_path(protocol, case, model, fold):
    return CACHE / 'task2' / protocol / case / model / f'fold_{fold:03}.npz'
