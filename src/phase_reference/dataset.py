"""Fixed publication splits. Truth is used here only for training/evaluation."""
import numpy as np
from .paths import ROOT
from .runtime import records, covariates, read

DATASETS = ['euler_original180', 'euler_independent72', 'sinusoidal36']


def select(dataset, split=None, limit=None):
    rr = records(ROOT / 'data/manifests' / (dataset + '.json'))
    if split:
        rr = [r for r in rr if r['split'] in split]
    return rr if limit is None else rr[:limit]


def load(dataset, rr):
    xx, yy, carriers, aa = [], [], [], []
    scale = read(ROOT / 'configs/calibration.json')['scale']
    for i, r in enumerate(rr):
        with np.load(ROOT / 'data/fields' / dataset / (r['geometry_id'] + '.npz'), allow_pickle=False) as z:
            a = {k: z[k] for k in ['eps', 'src_mask', 'x_um', 'y_um', 'source_center_um', 'Ez']}
        x, c, _ = covariates(r, a)
        xx.append(x); yy.append((a['Ez'][None] / scale).astype(np.complex64)); carriers.append(c); aa.append(a)
        if (i + 1) % 12 == 0:
            print('INPUTS', i + 1, len(rr), flush=True)
    return dict(x=np.stack(xx), y=np.stack(yy),
                c={m: np.stack([c[m] for c in carriers]) for m in carriers[0]}, arrays=aa, records=rr)
