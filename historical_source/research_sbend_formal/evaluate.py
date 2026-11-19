import ast,csv,time
import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal
from common import *
import model

def proxy_function():
    p=ROOT/'research_official_audit/validate_sparams.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    node=next(a for a in tree.body if isinstance(a,ast.FunctionDef) and a.name=='fd_directional_proxy')
    env={'np':np,'eigh_tridiagonal':eigh_tridiagonal};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),env)
    return env['fd_directional_proxy']
def writecsv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def field_metrics(pred,true,regions):
    err=abs(pred.astype(np.complex128)-true)**2;power=abs(true)**2
    result={'rel_l2':float(np.sqrt(err.sum()/power.sum()))}
    for name,mask in regions.items():
        den=power[mask].sum();assert den>0
        result[name+'_rel_l2']=float(np.sqrt(err[mask].sum()/den))
    return result
def coefficient_errors(a,b):
    return {'complex_error':float(abs(a-b)),'amplitude_error':float(abs(abs(a)-abs(b))),
        'squared_amplitude_error':float(abs(abs(a)**2-abs(b)**2))}
def summarize(rows):
    means=[];pairs=[];metrics=[k for k in rows[0] if k not in ['method','checkpoint','geometry_id','input_port','width_um','R_min_um']]
    for checkpoint in ['best','last']:
        for group in ['all','input1','input2']+['width_'+str(w*.025) for w in range(16,24)]+['R_3_5','R_5_7']:
            def ingroup(r):
                if group=='all':return True
                if group.startswith('input'):return r['input_port']==int(group[-1])
                if group.startswith('width'):return abs(r['width_um']-float(group[6:]))<1e-9
                return (r['R_min_um']<5) if group=='R_3_5' else (r['R_min_um']>=5)
            selected=[r for r in rows if r['checkpoint']==checkpoint and ingroup(r)]
            data={m:sorted([r for r in selected if r['method']==m],key=lambda r:r['geometry_id']) for m in METHODS}
            n=len(data['direct']);assert n>0 and all(len(a)==n for a in data.values())
            rng=np.random.default_rng(SEED);draw=rng.integers(0,n,(10000,n))
            for m,rr in data.items():
                means.append({'method':m,'checkpoint':checkpoint,'group':group,'n':n,
                    **{k:float(np.mean([a[k] for a in rr])) for k in metrics},
                    'rel_l2_median':float(np.median([a['rel_l2'] for a in rr])),
                    'rel_l2_p90':float(np.quantile([a['rel_l2'] for a in rr],.9))})
            for base in ['local_tangent','path_v1','linear','direct']:
                for metric in metrics:
                    a=np.array([r[metric] for r in data[base]]);b=np.array([r[metric] for r in data['path_v2']]);d=a-b
                    pairs.append({'checkpoint':checkpoint,'group':group,'metric':metric,'baseline':base,'candidate':'path_v2','n':n,
                        'baseline_mean':float(a.mean()),'path_v2_mean':float(b.mean()),'mean_baseline_minus_v2':float(d.mean()),
                        'relative_mean_error_reduction':float(1-b.mean()/a.mean()) if a.mean()!=0 else None,
                        'improved_geometries':int(np.sum(d>0)),'worsened_geometries':int(np.sum(d<0)),
                        'bootstrap_geometry_ci95':np.quantile(d[draw].mean(1),[.025,.975]).tolist()})
    return means,pairs

@torch.no_grad()
def run():
    verify_freeze();trained=read(OUT/'training_complete.json')
    for p,h in trained['weight_hashes'].items():assert sha(OUT/p)==h
    assert not (OUT/'evaluation_started.json').exists(),'One test event already started; do not repeat or delete marker'
    records=[r for r in read(OUT/'geometry_manifest.json')['records'] if r['split']=='test'];assert len(records)==100
    write('evaluation_started.json',{'time_utc':now(),'code_freeze_sha256':sha(OUT/'code_freeze.json'),
        'training_complete_sha256':sha(OUT/'training_complete.json'),'weight_hashes':trained['weight_hashes'],
        'one_event_includes_best_and_last':True})
    model.configure();cal=read(OUT/'calibration.json');proxy=proxy_function();truth=[];meta=[];regions=[];xx=[];cc={m:[] for m in METHODS}
    coefficients=[];truth_proxy=[];native=[]
    for r in records:
        with np.load(OUT/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
            xx.append(z['x']);regions.append({k[7:]:z[k] for k in z.files if k.startswith('region_')})
            for m in METHODS:cc[m].append(z['carrier_'+m])
        with np.load(OUT/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
            E=z['Ez'].astype(np.complex128);truth.append(E)
            args=(z['eps'],z['port_masks'],z['port_ids'],r['input_port'],1.55,.05);meta.append(args)
            c=proxy(E,*args);assert np.isfinite(c).all();truth_proxy.append(c);native.append(z['S_native'])
            for j in range(2):
                coefficients.append({'method':'truth_proxy','checkpoint':'none','geometry_id':r['geometry_id'],
                    'input_port':r['input_port'],'output_port':j+1,'is_reflection':j+1==r['input_port'],
                    'reference':'Meep_native','estimate_real':float(c[j].real),'estimate_imag':float(c[j].imag),
                    'target_real':float(z['S_native'][j].real),'target_imag':float(z['S_native'][j].imag),**coefficient_errors(c[j],z['S_native'][j])})
    xx=torch.from_numpy(np.stack(xx));cc={m:torch.from_numpy(np.stack(a)) for m,a in cc.items()}
    allrows=[];timings=[];prediction_hashes={};(OUT/'predictions').mkdir(exist_ok=True)
    for mname in METHODS:
        for checkpoint in ['best','last']:
            m=model.new();ck=torch.load(OUT/'runs'/mname/(checkpoint+'.pt'),map_location='cpu',weights_only=False);m.load_state_dict(ck['state']);m.eval();predictions=[];seconds=0
            for j in range(0,100,2):
                inp=xx[j:j+2].cuda();carrier=cc[mname][j:j+2].cuda();torch.cuda.synchronize();t=time.perf_counter()
                p=model.forward(m,inp,carrier);torch.cuda.synchronize();seconds+=time.perf_counter()-t
                predictions.extend((p[:,0]*cal['scale']).cpu().numpy())
            pred=np.stack(predictions);assert np.isfinite(pred).all()
            path=OUT/'predictions'/(mname+'_'+checkpoint+'.npz')
            np.savez_compressed(path,prediction=pred,geometry_id=np.array([r['geometry_id'] for r in records]),input_port=np.array([r['input_port'] for r in records]))
            prediction_hashes[str(path.relative_to(OUT))]=sha(path)
            timings.append({'method':mname,'checkpoint':checkpoint,'samples':100,'model_forward_recovery_seconds':seconds,'seconds_per_geometry':seconds/100})
            for i,r in enumerate(records):
                value=field_metrics(pred[i],truth[i],regions[i]);c=proxy(pred[i].astype(np.complex128),*meta[i]);assert np.isfinite(c).all()
                out=1 if r['input_port']==1 else 0;errors=coefficient_errors(c[out],truth_proxy[i][out])
                allrows.append({'method':mname,'checkpoint':checkpoint,'geometry_id':r['geometry_id'],'input_port':r['input_port'],
                    'width_um':r['params']['wg_width_um'],'R_min_um':r['params']['R_min_um'],**value,
                    **{'noninput_'+k:v for k,v in errors.items()}})
                for j in range(2):
                    for ref,target in [('truth_proxy',truth_proxy[i]),('Meep_native',native[i])]:
                        coefficients.append({'method':mname,'checkpoint':checkpoint,'geometry_id':r['geometry_id'],
                            'input_port':r['input_port'],'output_port':j+1,'is_reflection':j+1==r['input_port'],'reference':ref,
                            'estimate_real':float(c[j].real),'estimate_imag':float(c[j].imag),'target_real':float(target[j].real),
                            'target_imag':float(target[j].imag),**coefficient_errors(c[j],target[j])})
            print('EVALUATED '+mname+'/'+checkpoint,flush=True);del m,ck,p,pred;torch.cuda.empty_cache()
    summary,paired=summarize(allrows);writecsv('test_per_geometry.csv',allrows);writecsv('port_coefficients.csv',coefficients)
    write('test_summary.json',summary);write('paired_comparisons.json',paired);write('inference_costs.json',timings)
    write('evaluation_complete.json',{'time_utc':now(),'geometries':100,'models':10,'prediction_hashes':prediction_hashes,
        'metric_files_sha256':{n:sha(OUT/n) for n in ['test_per_geometry.csv','port_coefficients.csv','test_summary.json','paired_comparisons.json','inference_costs.json']},
        'one_test_event':True,'proxy_is_strict_S_parameter':False})
if __name__=='__main__':run()
