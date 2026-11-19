"""Scientific interface checks before full-resolution representation training."""
import numpy as np
import torch
import torch.nn.functional as F
from fourier_full import ROOT,configure,prepare,model_new,write,rel_mse
from official_bridge import wave_prior
from stage_a import prepare as prepare_v4


def main():
    configure();data=prepare();old=prepare_v4()
    x=data['x'];parts=[x[:,0].real]
    for j in range(1,10):
        z=x[:,1+(j-1)//2]
        parts.append(z.real if j%2 else z.imag)
    real=torch.stack(parts,1)
    down=F.interpolate(real,scale_factor=.5,mode='area')
    assert torch.equal(down,old['inputs'])
    assert data['meta']['calibration']==old['meta']['calibration']
    assert data['meta']['scale']==old['meta']['scale']
    assert torch.equal(data['c'][:,0],torch.tensor(old['carriers']['path']))
    assert x.shape==(72,6,160,480) and data['y'].shape==(72,1,160,480)
    assert real.shape==(72,10,160,480)
    assert float((data['c'].abs()-1).abs().max())<2e-7
    model=model_new(2026).eval()
    with torch.no_grad():
        src=x[:2].cuda();c=data['c'][:2].cuda();y=data['y'][:2].cuda()
        grid=model.observe_waveprior(src,torch.full((2,1),1.55,device='cuda'),torch.full((2,2),.05,device='cuda'))
        expected=wave_prior(data['eps'][:2]).cuda()
        prior_delta=float((grid-expected).abs().max())
        # Official forward reconstructs epsilon from normalized FP32 values;
        # accumulated phase rounding at column 479 can reach ~1.6e-5.
        assert torch.allclose(grid,expected,atol=3e-5),prior_delta
        p=torch.randn_like(y)
        physical=rel_mse(p*c,y);envelope=rel_mse(p,y/c)
        loss_delta=float((physical-envelope).abs().max())
        assert torch.allclose(physical,envelope,rtol=2e-6)
    write(ROOT/'interface_verification.json',dict(status='passed',input_resolution=[160,480],
        identical_base_features_to_v4_after_area_reduction=True,identical_calibration_and_carrier_to_v4=True,
        wave_prior_max_difference=prior_delta,unit_carrier_max_difference=float((data['c'].abs()-1).abs().max()),
        relative_mse_remodulation_equivalence_absolute_difference=loss_delta,
        shared_inputs=10,packed_complex_channels=6,stem_real_channels=16,test_evaluated=False))
    print('Full-resolution packing, identical v4 features/carrier, original wave prior, loss equivalence: passed.')


if __name__=='__main__':main()
