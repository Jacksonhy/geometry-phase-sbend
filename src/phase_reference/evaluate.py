"""Recompute physical-field and local phase metrics from published checkpoints."""
import argparse
import csv
from pathlib import Path
import numpy as np
from .paths import METHODS, SEEDS
from .dataset import DATASETS, select, load
from .runtime import load_net, infer_batch, write, sine_centerline
from . import metrics

_CONTEXT = {}


def measure(r, a, p):
    gid = r['geometry_id']
    if gid not in _CONTEXT:
        old = metrics.G['centerline_from_params']
        if 'length_um' in r['params']:
            metrics.G['centerline_from_params'] = sine_centerline
        try:
            poly, bend, masks = metrics.geometry(r, a)
        finally:
            metrics.G['centerline_from_params'] = old
        coords = metrics.paircoords(r, a, poly, bend, .5)
        _CONTEXT[gid] = (masks, coords)
    masks, (plus, minus, angle) = _CONTEXT[gid]
    e = a['Ez'].astype(np.complex128); p = p.astype(np.complex128)
    peak = float(abs(e[masks['bend_core']]).max())
    values = {}
    for region in ['whole', 'bend_core']:
        v = metrics.fieldmetric(p, e, masks[region], .05, peak)
        if v is None:
            raise ValueError('Too few valid field points')
        for key in ['complex_rel_l2', 'phase_wmae_deg']:
            values[region + '_' + key] = v[key]
    v = metrics.pairmetric(p, e, a, plus, minus, angle, .05, peak, None)
    if v is None:
        raise ValueError('Too few valid normal pairs')
    values.update({k: v[k] for k in ['normal_relative_phase_wmae_deg', 'coherence_complex_rel_l2']})
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, default='euler_independent72')
    p.add_argument('--split', nargs='+', help='For original180 use --split test')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    p.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    p.add_argument('--limit', type=int, help='Smoke check only; omitted for paper evaluation')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error('batch-size and limit must be positive')
    if args.dataset == 'euler_original180' and not args.split:
        args.split = ['test']
    rr = select(args.dataset, args.split, args.limit)
    if not rr:
        p.error('No records selected')
    d = load(args.dataset, rr)
    rows = []
    for seed in args.seeds:
        for method in args.methods:
            net = load_net(seed, method, args.device)
            for start in range(0, len(rr), args.batch_size):
                fields = infer_batch(net, d['x'][start:start+args.batch_size],
                                     d['c'][method][start:start+args.batch_size], args.device)
                for i, field in enumerate(fields, start):
                    rows.append(dict(geometry_id=rr[i]['geometry_id'], seed=seed, method=method,
                                     **measure(rr[i], d['arrays'][i], field)))
            del net
            print('SCORED', seed, method, len(rr), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'per_geometry.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = list(rows[0])[3:]
    summary = [dict(seed=s, method=m, n=len(rr), **{k: float(np.mean([r[k] for r in rows if r['seed']==s and r['method']==m])) for k in keys})
               for s in args.seeds for m in args.methods]
    write(args.output / 'summary.json', dict(dataset=args.dataset, gate=.05, normal_pair_eta=.5,
          limited_run=args.limit is not None, summaries=summary))


if __name__ == '__main__':
    main()
