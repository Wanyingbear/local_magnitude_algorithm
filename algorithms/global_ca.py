"""Compute global CA weighting features with dense Cholesky solves."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def global_ca(coordinates, *, alphas=(.3, .4, .5, .6, .7, .8)):
    """Return N-by-12 weights, ordered by p=1,2 and then alpha."""
    xyz = np.asarray(coordinates, dtype=float)
    a = np.asarray(alphas, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Expected finite nonempty (N,3) coordinates')
    if len(np.unique(xyz, axis=0)) != len(xyz):
        raise ValueError('Duplicate coordinates')
    if a.shape != (6,) or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('Supply six positive alpha values')
    distance = cdist(xyz, xyz)
    features = np.empty((len(xyz), 12))
    for pi, power in enumerate((1, 2)):
        for j, alpha in enumerate(a):
            kernel = np.exp(-alpha * distance**power)
            w = cho_solve(cho_factor(kernel, lower=True), np.ones(len(xyz)))
            if not np.isfinite(w).all() or np.max(np.abs(kernel @ w - 1)) > 1e-10:
                raise np.linalg.LinAlgError('Kernel solve residual exceeds tolerance')
            features[:, 6*pi + j] = w
    return features
