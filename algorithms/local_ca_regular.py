"""Uniform 12-feature local CA weighting: k=50, p=1,2, alpha=0.1,...,0.6."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

def local_ca_regular(coordinates, *, k=50, radius=None, alphas=(.1,.2,.3,.4,.5,.6), n_jobs=1):
    """Return (N,12) central weighting features; coordinates in angstroms.

    Only CA coordinates should be supplied. Neighborhoods include the center.
    radius overrides k; k>N uses all points. Columns: p=1 then p=2,
    each in supplied alpha order. No B-factor labels are used here.
    n_jobs controls Python worker threads over independent centers. When using
    multiple workers, limit BLAS to one thread externally with threadpoolctl
    to avoid nested thread pools. Small neighborhoods may be slower in threads.
    """
    xyz=np.asarray(coordinates,dtype=float);a=np.asarray(alphas,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Expected finite nonempty (N,3) coordinates')
    if len(np.unique(xyz,axis=0))!=len(xyz):raise ValueError('Duplicate coordinates')
    if a.shape!=(6,) or not np.isfinite(a).all() or np.any(a<=0):raise ValueError('Supply six positive alpha values')
    if radius is not None:
        if not np.isfinite(radius) or radius<=0:raise ValueError('radius must be positive')
    elif not isinstance(k,(int,np.integer)) or k<1:raise ValueError('k must be a positive integer')
    if not isinstance(n_jobs,(int,np.integer)) or n_jobs<1:raise ValueError('n_jobs must be a positive integer')
    tree=cKDTree(xyz);n=len(xyz);result=np.empty((n,12))
    def compute_center(center):
        bound=radius if radius is not None else float(np.max(tree.query(xyz[center],k=min(k,n))[0]))
        ids=np.asarray(tree.query_ball_point(xyz[center],np.nextafter(bound,np.inf)+1e-12),dtype=int)
        ids=ids[ids!=center];d=np.linalg.norm(xyz[ids]-xyz[center],axis=1)
        if radius is not None:ids=ids[d<=radius];d=d[d<=radius]
        ids=np.r_[center,ids[np.lexsort((ids,d))]]
        if radius is None:ids=ids[:min(k,n)]
        distance=cdist(xyz[ids],xyz[ids])
        for pi,p in enumerate((1,2)):
            for j,alpha in enumerate(a):
                kernel=np.exp(-alpha*distance**p)
                w=cho_solve(cho_factor(kernel,lower=True),np.ones(len(ids)))
                if not np.isfinite(w).all() or np.max(np.abs(kernel@w-1))>1e-10:
                    raise np.linalg.LinAlgError('Kernel solve residual exceeds tolerance')
                result[center,6*pi+j]=w[0]
    def compute_chunk(centers):
        for center in centers:
            compute_center(int(center))
    if n_jobs == 1:
        compute_chunk(range(n))
    else:
        # Each worker writes disjoint rows; map propagates worker exceptions.
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            list(executor.map(compute_chunk, np.array_split(np.arange(n), min(n_jobs,n))))
    return result
