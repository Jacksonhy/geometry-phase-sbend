"""Post-hoc target-spectrum diagnostic, never used to alter the frozen experiment.

This is not a network error bound: nonlinear stems and FFNs can create modes
outside the Fourier-layer truncation. Axis names follow image rows/columns.
"""
import json
import numpy as np
from fourier_full import ROOT,prepare,write


def main():
    data=prepare();y=data['y'].numpy()[:,0];c=data['c'].numpy()[:,0]
    result={}
    for variant,field in [('direct',y),('path',y/c)]:
        item={}
        for axis,modes,label in [(-2,40,'row_axis_modes40'),(-1,70,'column_axis_modes70')]:
            n=field.shape[axis]
            # Original rfft retains bins 0..modes-1 on real hidden channels.
            # Complex target diagnostics retain both signs up to that frequency.
            bins=np.fft.fftfreq(n)*n
            mask=abs(bins)<=modes-1+1e-5
            power=abs(np.fft.fft(field,axis=axis,norm='ortho'))**2
            keep=[slice(None)]*3;keep[axis]=mask
            fraction=power[tuple(keep)].sum((1,2))/power.sum((1,2))
            item[label]={split:dict(mean_fraction=float(fraction[ids].mean()),
                per_geometry=fraction[ids].tolist()) for split,ids in [('train',data['tr']),('val',data['va'])]}
        result[variant]=item
    report=dict(status='diagnostic only; not used for training or selection',
        convention='full complex target FFT; retain symmetric +/- (modes-1) bins; no taper/window',
        limitation='Target spectrum only, not hidden-channel spectrum or network expressivity/error bound; nonlinear blocks can generate higher frequencies.',
        test_evaluated=False,results=result)
    write(ROOT/'spectral_diagnostic.json',report)
    print(json.dumps({v:{axis:{s:r['mean_fraction'] for s,r in splits.items()} for axis,splits in item.items()} for v,item in result.items()},indent=2))


if __name__=='__main__':main()
