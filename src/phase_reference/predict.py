"""Geometry-only command line inference; preparation never advances Meep time."""
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
from .paths import ROOT, METHODS, SEEDS
from .runtime import (records, read, write, sha, material, save_input,
                      field_free_guard, load_net, infer_batch, identity)


def prepare(rr, out):
    out.mkdir(parents=True, exist_ok=True)
    for r in rr:
        dest = out / (r['geometry_id'] + '.npz')
        if dest.exists():
            raise FileExistsError(dest)
        save_input(dest, r, material(r))
        print('PREPARED', r['geometry_id'], flush=True)
    write(out / 'preparation.json', dict(records=rr, simulation_time=0,
          FDTD_runs=0, truth_files_read=0,
          calibration_sha256=sha(ROOT / 'configs/calibration.json')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry-json', required=True, type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--prepared-inputs', type=Path)
    p.add_argument('--meep-python', help='Python executable with Meep, for automatic preparation')
    p.add_argument('--method', choices=METHODS, default='local_tangent')
    p.add_argument('--seed', type=int, choices=SEEDS, default=2026)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('--batch-size must be positive')
    field_free_guard()
    rr = records(a.geometry_json)
    if a.prepare_only:
        prepare(rr, a.output)
        return
    if a.prepared_inputs is None:
        if not a.meep_python:
            p.error('Supply --prepared-inputs or --meep-python')
        a.prepared_inputs = a.output / 'inputs'
        subprocess.run([a.meep_python, '-m', 'phase_reference.predict',
                        '--geometry-json', str(a.geometry_json.resolve()), '--prepare-only',
                        '--output', str(a.prepared_inputs.resolve())], check=True)
    contract = read(a.prepared_inputs / 'preparation.json')
    saved = {r['geometry_id']: r for r in contract['records']}
    for r in rr:
        if r['geometry_id'] not in saved or identity(r) != identity(saved[r['geometry_id']]):
            raise ValueError('Prepared geometry or excitation differs from request')
    if contract['calibration_sha256'] != sha(ROOT / 'configs/calibration.json'):
        raise ValueError('Prepared calibration differs from training calibration')
    net = load_net(a.seed, a.method, a.device)
    a.output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rr), a.batch_size):
        batch = rr[start:start + a.batch_size]
        xx, cc, grids = [], [], []
        for r in batch:
            with np.load(a.prepared_inputs / (r['geometry_id'] + '.npz'), allow_pickle=False) as z:
                xx.append(z['x']); cc.append(z['carrier_' + a.method])
                grids.append((z['x_um'], z['y_um']))
        pred = infer_batch(net, xx, cc, a.device)
        for r, field, (x, y) in zip(batch, pred, grids):
            np.savez_compressed(a.output / (r['geometry_id'] + '.npz'), Ez=field,
                                x_um=x, y_um=y, method=a.method, seed=a.seed)
    write(a.output / 'inference.json', dict(records=rr, method=a.method, seed=a.seed,
          FDTD_runs=0, truth_files_read=0, calibration_sha256=contract['calibration_sha256']))
    print('INFERENCE_COMPLETE', len(rr), flush=True)


if __name__ == '__main__':
    main()
