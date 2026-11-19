"""Frozen downstream five-line extractor, exclusively new training data."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from common import OUT,read,write,sha,v1

def extract(E,x,y,params,input_port):
    poly,bend=v1.centerline_from_params(params);w=params['wg_width_um']
    lo,hi=(-10.5,bend[0,0]-.75) if input_port==2 else (bend[-1,0]+.75,10.5)
    xs=x[(x>=lo)&(x<=hi)]
    if len(xs)<20:raise RuntimeError('beta ROI fewer than 20 samples')
    yc=bend[0 if input_port==2 else -1,1];ns=np.linspace(-.25*w,.25*w,5)
    pts=np.stack(np.meshgrid(xs,ns,indexing='xy'),axis=-1);pts[...,1]+=yc
    interp=RegularGridInterpolator((y,x),E,bounds_error=True)
    z=interp(pts[...,[1,0]]).mean(0);phase=np.unwrap(np.angle(z));fit=np.polyfit(xs,phase,1)
    beta=float((1 if input_port==1 else -1)*fit[0])
    if not np.isfinite(beta) or beta<=0:raise RuntimeError('Nonpositive beta')
    return {'beta_rad_um':beta,'count':len(xs),'x_bounds_um':[float(xs.min()),float(xs.max())],
        'phase_fit_rms':float(np.sqrt(np.mean((phase-np.polyval(fit,xs))**2)))}

def fit_training(records):
    assert len(records)==64 and all(r['split']=='train' for r in records)
    if (OUT/'calibration.json').exists():raise RuntimeError('Calibration already performed')
    rows=[];energy=0;count=0
    for r in records:
        p=OUT/'data'/r['geometry_id']/'sample.npz'
        with np.load(p,allow_pickle=False) as z:
            assert str(z['split'])=='train' and str(z['geometry_id'])==r['geometry_id']
            E=z['Ez'];q=extract(E,z['x_um'],z['y_um'],r['params'],r['input_port'])
            rows.append({'geometry_id':r['geometry_id'],'sample_sha256':sha(p),'eps_max':float(z['eps'].max()),**q})
            energy+=float(np.sum(abs(E)**2));count+=E.size
    xs=np.array([a['eps_max'] for a in rows]);ys=np.array([a['beta_rad_um'] for a in rows]);coef=np.polyfit(xs,ys,1)
    rmse=float(np.sqrt(np.mean(((np.polyval(coef,xs)-ys)/ys)**2)))
    passed=rmse<=.01 and bool(np.all(np.polyval(coef,xs)>0))
    result={'time_utc':__import__('common').now(),'coefficients_rad_um':coef.tolist(),'relative_rmse':rmse,
       'scale':float(np.sqrt(energy/count)),'rows':rows,'passed':passed,'train_only':True,'validation_or_test_used':False}
    write('calibration.json',result)
    if not passed:raise RuntimeError('Training beta calibration failed fixed 1% gate; no search allowed')
    return result
