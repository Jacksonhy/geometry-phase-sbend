"""Verify linear carrier before training; labels not used to construct carrier."""
import sys
import numpy as np
import torch
from run_linear import ROOT,V5,ff,prepare,forward,ORIGINAL_FORWARD,check_existing
sys.path.insert(0,str(ROOT.parent/'research_v4'))
from stage_a import prepare as v4_prepare


def main():
    ff.configure();check_existing();data=prepare();old=v4_prepare()
    assert np.array_equal(data['c'][:,0].numpy(),old['carriers']['linear'])
    assert np.array_equal(data['path_c'][:,0].numpy(),old['carriers']['path'])
    assert data['meta']['calibration']==old['meta']['calibration']
    m=ff.model_new(2026).eval()
    with torch.no_grad():
        x=data['x'][:2].cuda();c=data['c'][:2].cuda()
        plain=ORIGINAL_FORWARD(m,x,c,'direct','fp32')
        result=forward(m,x,c,'linear','fp32')
        assert torch.equal(result,plain*c)
        y=data['y'][:2].cuda()
        assert torch.allclose(ff.rel_mse(result,y),ff.rel_mse(plain,y/c),rtol=2e-6)
    row=dict(status='passed',linear_carrier_identical_to_v4=True,path_carrier_identical_to_v4=True,
        forward_equals_official_output_times_linear_carrier=True,physical_loss_equivalence=True,
        resolution=[160,480],initial_state_sha256=ff.state_hash(m),test_evaluated=False)
    baseline=ff.read(V5/'runs/direct_lr0.002_seed2026/summary.json')
    assert row['initial_state_sha256']==baseline['initial_state_sha256']
    ff.write(ROOT/'interface_verification.json',row);print(row)


if __name__=='__main__':main()
