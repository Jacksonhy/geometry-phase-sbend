"""Simple label-budget baselines and phase-carrier diagnostics."""
import json
import numpy as np
import torch
from experiment import ROOT, load_data, metric_arrays, geometry_features, write_json
from evaluate_fresh import inference_inputs


def aggregate(pred,target,ports):
    m=metric_arrays(pred,target)
    overall={k:float(v.mean()) for k,v in m.items()}
    if not np.any(pred):
        overall['intensity_weighted_phase_mae']=None  # Phase of an identically zero field is undefined.
    return dict(overall=overall,
                port_macro_rel_l2=float(np.mean([m['rel_l2'][ports==p].mean() for p in np.unique(ports)])),
                per_sample_rel_l2=m['rel_l2'].tolist())


def baseline_main():
    torch.set_num_threads(4)
    with np.load(ROOT/'fresh_test.npz',allow_pickle=False) as z: fresh={k:z[k] for k in z.files}
    truth=fresh['Ez_real']+1j*fresh['Ez_imag']
    results={}
    for n in [32,64]:
        old,e,x,carriers,tr,va,te,scale,audit=load_data(n)
        eps=old['eps'][tr]; src=old['src_mask'][tr]; ports=old['input_port'][tr]
        # Match exactly the epsilon and source scaling used for the neural inputs.
        distance=((fresh['eps'][:,None]-eps[None])/4.2)**2+(fresh['src_mask'][:,None]-src[None])**2
        distance=distance.mean(axis=(2,3))
        distance[fresh['input_port'][:,None]!=ports[None]]=np.inf
        nearest=np.argmin(distance,axis=1)
        results[f'n{n}']={'zero':aggregate(np.zeros_like(truth),truth,fresh['input_port'])}
        for mode in ['raw','linear','path']:
            _,fresh_carrier,_=inference_inputs(fresh['eps'],fresh['src_mask'],audit['calibration_coefficients'],mode)
            train_envelope=e[tr]/carriers[mode][tr]
            mean=np.stack([train_envelope[ports==p].mean(axis=0) for p in fresh['input_port']])
            nn=train_envelope[nearest]
            for name,pred in [('port_mean',mean*fresh_carrier),('port_nearest_geometry',nn*fresh_carrier)]:
                result=aggregate(pred,truth,fresh['input_port'])
                if name=='port_nearest_geometry': result['train_indices']=tr[nearest].tolist()
                results[f'n{n}'][f'{name}_{mode}']=result
        print('baselines',n,{k:round(v['overall']['rel_l2'],4) for k,v in results[f'n{n}'].items()},flush=True)
    write_json(ROOT/'fresh_evaluation/baselines.json',results)


def phase_diagnostics():
    torch.set_num_threads(4)
    old,e,x,carriers,tr,va,te,scale,audit=load_data(32)
    with np.load(ROOT/'fresh_test.npz',allow_pickle=False) as z: fresh={k:z[k] for k in z.files}
    result={}
    for name,eps,src,field in [('train32',old['eps'][tr],old['src_mask'][tr],e[tr]),
                               ('fresh37',fresh['eps'],fresh['src_mask'],fresh['Ez_real']+1j*fresh['Ez_imag'])]:
        k=np.polyval(audit['calibration_coefficients'],eps.max(axis=(1,2)))
        rows=[]
        for i,(a,b) in enumerate(zip(eps,src)):
            f=geometry_features(a,b)
            branch=0 if f['sy']<a.shape[0]/2 else 1
            # For left injection inspect both output branches. For right injection inspect the excited branch.
            for branch in ([0,1] if f['direction']==1 else [branch]):
                center=f['upper'] if branch==0 else f['lower']
                xx=np.arange(a.shape[1]); y=np.rint(center).astype(int)
                path_len=np.sqrt(1+np.diff(center)**2)
                ff=field[i,y,xx]
                strength=abs(ff[:-1])*abs(ff[1:])
                slope=np.angle(ff[1:]*np.conj(ff[:-1]))
                for region,mask in [('trunk',(xx[:-1]>30)&(xx[:-1]<f['junction']-10)),
                                    ('bend',(xx[:-1]>f['junction']+10)&(abs(np.diff(center))>.03))]:
                    if mask.sum()<5: continue
                    row=dict(index=i,branch=branch,region=region)
                    for mode in ['raw','linear','path']:
                        if mode=='raw': increment=np.zeros_like(slope)
                        else:
                            phi=k[i]*f[mode][y,xx]
                            increment=np.diff(phi)
                        residual=np.angle(np.exp(1j*(slope-increment)))/path_len
                        row[mode+'_rms_rad_per_pixel']=float(np.sqrt(np.sum(strength[mask]*residual[mask]**2)/np.sum(strength[mask])))
                    rows.append(row)
        result[name]=dict(rows=rows,summary={region:{mode:float(np.mean([r[mode+'_rms_rad_per_pixel'] for r in rows if r['region']==region]))
                         for mode in ['raw','linear','path']} for region in ['trunk','bend']})
    write_json(ROOT/'fresh_evaluation/phase_diagnostics.json',result)
    print('Phase diagnostics',{k:v['summary'] for k,v in result.items()},flush=True)


if __name__=='__main__':
    baseline_main()
    phase_diagnostics()
