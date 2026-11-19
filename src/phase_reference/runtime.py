"""Portable geometry-only preprocessing using the frozen numerical functions."""
import contextlib
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from .paths import ROOT, PICFLOW, METHODS
from . import features as F


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def records(path):
    obj = read(path)
    rr = obj.get('records', [obj]) if isinstance(obj, dict) else obj
    for i, r in enumerate(rr):
        r.setdefault('geometry_id', f'query_{i:04d}')
        gid = r['geometry_id']
        if not isinstance(gid, str) or gid in {'.', '..'} or '/' in gid or '\\' in gid:
            raise ValueError('geometry_id must be a safe basename')
        if r['input_port'] not in [1, 2]:
            raise ValueError('input_port must be 1 or 2')
        p = r['params']
        if not all(np.isfinite(v) for v in p.values()):
            raise ValueError('Parameters must be finite')
        if 'length_um' not in p:
            for k, lo, hi in [('wg_width_um', .4, .575), ('lateral_offset_um', 2., 5.5), ('R_min_um', 3., 7.)]:
                if not lo - 1e-10 <= p[k] <= hi + 1e-10:
                    raise ValueError(f'{k} outside evaluated Euler range')
                if abs(p[k] / .025 - round(p[k] / .025)) > 1e-8:
                    raise ValueError('Euler parameters must use the evaluated 0.025 um lattice')
        elif p['length_um'] <= 0:
            raise ValueError('Sinusoidal length must be positive')
    if len({r['geometry_id'] for r in rr}) != len(rr):
        raise ValueError('Duplicate geometry_id')
    return rr


def identity(r):
    return r['params'], r['input_port']


def sine_points(offset, length, n_pts=256):
    t = np.linspace(0., 1., n_pts)
    return length * (t - .5), offset * (t - np.sin(2 * np.pi * t) / (2 * np.pi) - .5)


def sine_centerline(params, x_bounds=(-12., 12.)):
    x, y = sine_points(params['lateral_offset_um'], params['length_um'])
    bend = np.column_stack([x, y])
    if not x_bounds[0] < x[0] < x[-1] < x_bounds[1]:
        raise ValueError('Sinusoidal bend does not fit the domain')
    return np.vstack([[x_bounds[0], y[0]], bend, [x_bounds[1], y[-1]]]), bend


@contextlib.contextmanager
def family_geometry(record):
    old = F.v1.centerline_from_params
    if 'length_um' in record['params']:
        F.v1.centerline_from_params = sine_centerline
    try:
        yield
    finally:
        F.v1.centerline_from_params = old


def covariates(record, a):
    """Only material/source/geometry keys are accessed; no query field."""
    cal = read(ROOT / 'configs/calibration.json')
    beta = float(np.polyval(cal['coefficients_rad_um'], float(a['eps'].max())))
    with family_geometry(record):
        distances, _, f = F.geometry(record['params'], a['x_um'], a['y_um'], a['source_center_um'], record['input_port'])
        x, carriers = F.inputs(a['eps'], a['src_mask'], a['x_um'], a['y_um'], a['source_center_um'], record['input_port'], beta, distances)
        poly = f['polyline_um']
        _, _, arc, _ = F.v1.curve_info(poly)
        ss = float(F.v1.project_points(a['source_center_um'][None, :2], poly)['s'][0])
        da = (1 if record['input_port'] == 1 else -1) * np.broadcast_to(np.interp(a['x_um'], poly[:, 0], arc) - ss, a['eps'].shape)
        carriers['arc_only'] = np.exp(1j * beta * da).astype(np.complex64)[None]
    return x, {m: carriers[m] for m in METHODS}, beta


def material(record):
    """Meep initialization at t=0, exactly as the historical deployment."""
    sys.path.insert(0, str(PICFLOW / 'FDTD'))
    from sbend import sbend as module
    import meep as mp
    mp.verbosity(0)
    def forbidden(*args, **kwargs):
        raise RuntimeError('FDTD time propagation is forbidden during input preparation')
    mp.Simulation.run = forbidden
    mp.Simulation.step = forbidden
    p = record['params']
    original = module.euler_sbend_points
    if 'length_um' in p:
        module.euler_sbend_points = lambda lateral_offset_um, R_min_um, n_pts=256: sine_points(lateral_offset_um, p['length_um'], n_pts)
    try:
        dev = module.EulerSBend2D(wg_width=p['wg_width_um'], lateral_offset_um=p['lateral_offset_um'],
                                 R_min_um=p.get('R_min_um', 5.), wavelength_um=1.55,
                                 resolution=20, dpml=1, cell_x=26, cell_y=10)
    finally:
        module.euler_sbend_points = original
    sim, _, dft, _ = dev._build_sim_single(input_port=record['input_port'])
    sim.init_sim()
    if sim.meep_time() != 0:
        raise RuntimeError('Material preparation advanced simulation time')
    eps = sim.get_epsilon().T[20:-20, 20:-20].astype(np.float32)
    meta = sim.get_array_metadata(dft_cell=dft)
    x, y = np.array(meta.x[20:-20]), np.array(meta.y[20:-20])
    src = sim.sources[0]
    source = np.array([src.center.x, src.center.y, src.center.z])
    half = float(src.size.y) / 2
    X, Y = np.meshgrid(x, y)
    mask = ((X - source[0])**2 + (Y - np.clip(Y, source[1] - half, source[1] + half))**2 <= .15**2 + 1e-14).astype(np.float32)
    sim.reset_meep()
    if eps.shape != mask.shape or eps.shape != (160, 480):
        raise ValueError('Unexpected material/source grid')
    return dict(eps=eps, src_mask=mask, x_um=x, y_um=y, source_center_um=source)


def save_input(path, record, arrays):
    x, carriers, beta = covariates(record, arrays)
    np.savez_compressed(path, x=x, **{'carrier_' + m: carriers[m] for m in METHODS},
                        **arrays, geometry_id=np.array(record['geometry_id']),
                        input_port=np.int32(record['input_port']), beta_rad_um=beta)


def field_free_guard():
    """Protect all published truth fields regardless of checkout location."""
    truth_root = (ROOT / 'data/fields').resolve()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved.is_relative_to(truth_root) or path.name == 'sample.npz' or 'predictions' in path.parts:
            raise RuntimeError('Geometry-only prediction blocked a truth/prediction path')
    sys.addaudithook(audit)


def load_net(seed, method, device='cuda'):
    import torch
    from . import model
    row = next(r for r in read(ROOT / 'configs/checkpoints.json') if r['seed'] == seed and r['method'] == method)
    path = ROOT / row['path']
    if sha(path) != row['sha256']:
        raise ValueError('Checkpoint SHA-256 mismatch')
    model.configure()
    net = model.new(seed=seed, device=device)
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['seed'] != seed or ck['method'] != method:
        raise ValueError('Checkpoint identity mismatch')
    net.load_state_dict(ck['state'])
    return net.eval()


def infer_batch(net, arrays, carriers, device='cuda'):
    import torch
    from . import model
    with torch.inference_mode():
        x = torch.from_numpy(np.stack(arrays)).to(device)
        c = torch.from_numpy(np.stack(carriers)).to(device)
        pred = model.forward(net, x, c)[:, 0] * read(ROOT / 'configs/calibration.json')['scale']
    return pred.cpu().numpy()
