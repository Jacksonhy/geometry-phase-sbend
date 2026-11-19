"""Fresh training on the fixed 64/16 split; test fields are never opened."""
import argparse
from pathlib import Path
import numpy as np
import torch
from . import model
from .paths import METHODS, SEEDS
from .dataset import select, load
from .runtime import write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, default=2026)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=2400, help='Use 2400 for the paper budget')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if not 1 <= a.steps <= 2400:
        p.error('steps must be 1..2400')
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use an empty output directory for a fresh run')
    a.output.mkdir(parents=True, exist_ok=True)
    rr = select('euler_original180', ['train', 'val'])
    rr.sort(key=lambda r: 0 if r['split'] == 'train' else 1)
    assert len(rr) == 80 and sum(r['split'] == 'train' for r in rr) == 64
    d = load('euler_original180', rr)
    x, y, c = (torch.from_numpy(d[k] if k != 'c' else d[k][a.method]) for k in ['x', 'y', 'c'])
    model.configure(); net = model.new(a.seed, a.device)
    initial = model.state_hash(net)
    batches = model.batches()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2400, eta_min=.0001)
    history, best = [], float('inf')
    for step in range(1, a.steps + 1):
        ids = batches[step - 1]
        net.train(); optimizer.zero_grad(set_to_none=True)
        pred = model.forward(net, x[ids].to(a.device), c[ids].to(a.device))
        loss = model.rel_mse(pred, y[ids].to(a.device)).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        if not torch.isfinite(grad):
            raise RuntimeError('Nonfinite gradient')
        optimizer.step(); scheduler.step()
        if step == 1 or step % 50 == 0:
            net.eval(); vals = []
            with torch.inference_mode():
                for j in range(64, 80, 2):
                    pp = model.forward(net, x[j:j+2].to(a.device), c[j:j+2].to(a.device))
                    vals.extend(model.rel_mse(pp, y[j:j+2].to(a.device)).sqrt().cpu().tolist())
            val = float(np.mean(vals))
            history.append(dict(step=step, loss=float(loss.detach()), val_rel_l2=val))
            print(history[-1], flush=True)
            if val < best:
                best = val
                torch.save(dict(state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                                method=a.method, seed=a.seed, step=step, val_rel_l2=val,
                                initial_state_sha256=initial, config=model.CONFIG), a.output / 'best.pt')
    write(a.output / 'history.json', history)
    write(a.output / 'run.json', dict(method=a.method, seed=a.seed, updates=a.steps,
          full_paper_budget=a.steps == 2400, initial_state_sha256=initial,
          split_counts=dict(train=64, val=16), test_fields_read=0))


if __name__ == '__main__':
    main()
