"""Isolated Y BranchTangent pilot; no historical writes or target-driven carrier."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,csv,hashlib,time,random,itertools,argparse,traceback
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates
BASE=Path('/mnt/d/game');OUT=Path(__file__).resolve().parent;OLD=BASE/'research_e3_2400'
sys.path.insert(0,str(OLD))
import train as stage
ff=stage.ff
from experiment import metric_arrays
METHODS=['linear','path','branch_tangent']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def csvwrite(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def srcval(a,f):
    return float(map_coordinates(np.asarray(a,dtype=np.float64),np.array([[f['sy']],[f['sx']]]),order=1,mode='nearest')[0])
def correction(f):
    h,w=f['core'].shape;yy,xx=np.mgrid[:h,:w]
    centers=np.array([f['upper'],f['lower']],dtype=np.float64)
    branch=(yy>(centers[0]+centers[1])[None,:]/2).astype(int)
    slopes=np.gradient(centers,axis=1,edge_order=1);ty=slopes/np.sqrt(1+slopes**2)
    q=np.full((h,w),f['direction'],dtype=np.float64)
    sb=0 if f['sy']<h/2 else 1
    other=(branch!=sb)&(xx>f['junction']) if f['direction']==-1 else np.zeros((h,w),bool)
    q[other]=1.
    raw=q*ty[branch,xx]*(yy-centers[branch,xx]);offset=srcval(raw,f);u=raw-offset
    assert abs(srcval(u,f))<1e-10
    assert np.all(q[other]==1) and np.all(q[~other]==f['direction'])
    return u,dict(branch=branch,q=q,ty=ty,centers=centers,source_offset_pixels=offset)
def carrier(ep,sm,cal,gid,port):
    f=ff.geometry_features(ep,sm);u,info=correction(f);k=float(np.polyval(cal,float(ep.max())))
    assert f['direction']==(1 if port==1 else -1)
    distance=f['path'].astype(np.float64)+u
    c=np.exp(1j*k*distance).astype(np.complex64)
    assert np.isfinite(c).all() and np.max(abs(abs(c)-1))<2e-7
    old=np.exp(1j*k*f['path']).astype(np.complex64)
    seam=(info['branch'][1:]!=info['branch'][:-1])
    core=f['core'][1:]|f['core'][:-1]
    jumps={}
    for label,a in [('path',old),('branch_tangent',c)]:
        vals=abs(a[1:]-a[:-1])[seam];cv=abs(a[1:]-a[:-1])[seam&core]
        jumps[label+'_seam_carrier_jump_max']=float(vals.max()) if len(vals) else 0.
        jumps[label+'_seam_core_carrier_jump_max']=float(cv.max()) if len(cv) else 0.
    rec=dict(geometry_id=str(gid),input_port=int(port),junction_column=int(f['junction']),
        source_correction_pixels=srcval(u,f),raw_source_offset_pixels=info['source_offset_pixels'],
        max_abs_correction_pixels=float(abs(u).max()),max_abs_correction_phase_rad=float(abs(k*u).max()),
        nonzero_source_slope=float(np.interp(f['sx'],np.arange(ep.shape[1]),info['ty'][0 if f['sy']<ep.shape[0]/2 else 1])),
        carrier_unit_modulus_error=float(np.max(abs(abs(c)-1))),**jumps)
    return c,rec
def oracle():
    # Exact independently specified linear branch slopes; right opposite branch must turn outward.
    h,w=24,30;xx=np.arange(w,dtype=float)
    centers=np.array([8-.15*(xx-10),16+.25*(xx-10)])
    checks=[]
    for port,sy,direction in [(1,12,1),(2,5,-1),(3,20,-1)]:
        f=dict(core=np.ones((h,w),bool),upper=centers[0],lower=centers[1],sx=2.3 if port==1 else 26.2,
               sy=float(sy),direction=direction,junction=10)
        u,info=correction(f)
        raw=np.empty((h,w))
        for y in range(h):
            for x in range(w):
                b=int(y>(centers[0,x]+centers[1,x])/2)
                slope=[-.15,.25][b];q=direction
                if direction==-1 and b!=(0 if sy<h/2 else 1) and x>10:q=1
                raw[y,x]=q*slope/np.sqrt(1+slope*slope)*(y-centers[b,x])
        expected=raw-srcval(raw,f)
        err=float(abs(u-expected).max());assert err<1e-12
        assert abs(srcval(u,f))<1e-12
        checks.append(dict(port=port,max_error_pixels=err,source_correction=srcval(u,f)))
    # Zero transverse slopes leave historical arc route exactly unchanged.
    f.update(upper=np.full(w,8.),lower=np.full(w,16.));u,_=correction(f);assert np.all(u==0)
    return checks
def protected():
    paths=set()
    for folder in ['research_v3','research_v4','research_v5','research_e1','research_e3_2400','research_m4']:
        for p in (BASE/folder).glob('*.py'):paths.add(p)
        for p in (BASE/folder).glob('*.json'):paths.add(p)
    for n in read(OLD/'manifest.json')['sources']:paths.add(BASE/n)
    for n in read(OLD/'manifest.json')['inherited_sources']:paths.add(BASE/n)
    paths.add(BASE/'picflow_ybranch_smoke_80.npz')
    for ds in ['research_e2','research_m4']:
        for p in (BASE/ds).glob('*'):
            if p.is_file() and p.suffix in ['.py','.json','.npz','.csv','.md']:paths.add(p)
        paths.update((BASE/ds/'predictions').glob('*.npz'))
    for m in ['linear','path']:
        paths.update((OLD/'runs'/f'{m}_steps2400_seed2026').glob('*'))
    return {str(p.relative_to(BASE)):sha(p) for p in sorted(paths) if p.is_file()}
def verify():
    fr=read(OUT/'freeze.json')
    for n,h in fr['protected_hashes'].items():assert sha(BASE/n)==h,'Protected file changed: '+n
    for n,h in fr['source_hashes'].items():assert sha(OUT/n)==h,'Frozen source changed: '+n
def prepare():
    stage.setup();data=stage.prepare();frozen=read(OLD/'manifest.json')
    for k in ['calibration','scale','train_geometry_ids','val_geometry_ids','resolution']:
        assert data['meta'][k]==frozen[k],k
    with np.load(BASE/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as z:
        keep=np.flatnonzero(z['split']!='test');ids=z['geometry_id'][keep];ports=z['input_port'][keep]
    cs=[];records=[]
    for ep,sm,gid,port in zip(data['eps'],data['src'],ids,ports):
        c,r=carrier(ep,sm,data['meta']['calibration'],gid,int(port));cs.append(c);records.append(r)
    data['bt_c']=torch.from_numpy(np.stack(cs)[:,None])
    data['ids']=ids;data['ports']=ports
    return data,records
def bind():
    # Do not alter source files: adapt only the in-memory historical loop endpoints.
    original=stage.carrier
    stage.carrier=lambda data,variant:data['bt_c'] if variant=='branch_tangent' else original(data,variant)
    previous=ff.forward
    ff.forward=lambda model,x,c,variant,precision:stage.e1.ORIGINAL_FORWARD(model,x,c,'path',precision) if variant=='branch_tangent' else previous(model,x,c,variant,precision)
    stage.ROOT=OUT
    stage.prior_summary=lambda variant:read(BASE/'research_v5/runs/path_lr0.002_seed2026/summary.json')
def preflight(data,records):
    assert not (OUT/'freeze.json').exists(),'Already frozen'
    stage.check_old();settings=read(OLD/'manifest.json')
    checks=oracle();replays=[];ff.configure()
    rng=np.random.default_rng(2026);batch=np.stack([rng.choice(data['tr'],2,replace=False) for _ in range(2400)])
    bh=hashlib.sha256(batch.tobytes()).hexdigest()
    for method in ['linear','path']:
        root=OLD/'runs'/f'{method}_steps2400_seed2026';s=read(root/'summary.json')
        random.seed(2026);m=ff.model_new(2026);initial=ff.state_hash(m)
        assert initial==s['initial_state_sha256'] and bh==s['batches_sha256']
        for ckname in ['best','last']:
            assert sha(root/(ckname+'.pt'))==s[ckname+'_sha256']
            ck=torch.load(root/(ckname+'.pt'),map_location='cpu',weights_only=False)
            assert ck['initial_state_sha256']==initial and ck['batches_sha256']==bh
            for key in ['calibration','scale']:assert ck[key]==data['meta'][key]
            assert ck['config']==ff.CONFIG
            m.load_state_dict(ck['state'])
            actual=stage.score(m,data,data['va'],method)
            expected=np.array(s['best_val_per_sample' if ckname=='best' else 'final_val_per_sample'])
            err=float(abs(actual-expected).max());assert err<1e-7
            replays.append(dict(method=method,checkpoint=ckname,max_per_geometry_difference=err,mean_rel_l2=float(actual.mean())))
        del m,ck;torch.cuda.empty_cache()
    # Check fresh inference inputs/carriers match historical training preparation bit for bit.
    x,pc=ff.infer_inputs(data['eps'],data['src'],data['meta']['calibration'])
    assert torch.equal(x,data['x']) and torch.equal(pc,data['path_c']);del x,pc
    bind();random.seed(2026);m=ff.model_new(2026);initial=ff.state_hash(m);m.train()
    p=ff.forward(m,data['x'][:2].cuda(),data['bt_c'][:2].cuda(),'branch_tangent','fp32')
    loss=ff.rel_mse(p,data['y'][:2].cuda()).mean();loss.backward()
    assert torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in m.parameters())
    real=sum(q.numel()*(2 if q.is_complex() else 1) for q in m.parameters());assert real==2944738
    del m,p,loss;torch.cuda.empty_cache()
    save('geometry_checks_trainval.json',records)
    save('preflight.json',dict(passed=True,optimizer_updates=0,no_test_fields=True,synthetic_oracle=checks,
        historical_replays=replays,inputs_path_carrier_bit_exact=True,initial_state_sha256=initial,batches_sha256=bh,
        real_scalar_degrees=real,runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))
    save('freeze.json',dict(time_utc=time.time(),seed=2026,methods=METHODS,protected_hashes=protected(),
        source_hashes={n:sha(OUT/n) for n in ['pilot.py','PROTOCOL.md']},
        preflight_sha256=sha(OUT/'preflight.json'),prior_test_use_disclosed=True))
    print('PREFLIGHT PASSED; carrier routing and historical validation replays verified',flush=True)
def train_bt(data):
    verify();bind();ff.configure()
    result=stage.train_one(data,'branch_tangent')
    pre=read(OUT/'preflight.json')
    assert result['initial_state_sha256']==pre['initial_state_sha256'] and result['batches_sha256']==pre['batches_sha256']
    # Historical loop already scored final and best training. Replay best validation explicitly.
    ck=torch.load(OUT/'runs/branch_tangent_steps2400_seed2026/best.pt',map_location='cpu',weights_only=False)
    m=ff.model_new(2026);m.load_state_dict(ck['state'])
    actual=stage.score(m,data,data['va'],'branch_tangent');err=float(abs(actual-np.array(result['best_val_per_sample'])).max())
    assert err<1e-7
    save('training_complete.json',dict(seed=2026,result=result,best_validation_replay_max_error=err,
        freeze_sha256=sha(OUT/'freeze.json'),new_models=1,other_methods_reused=True))
    print('TRAINING COMPLETE',flush=True)
def load_benchmark(ds):
    folder=BASE/('research_e2' if ds=='E2' else 'research_m4')
    with np.load(folder/'test_materials.npz',allow_pickle=False) as z:
        ep=z['eps'];sm=z['src_mask'];ids=z['geometry_id'];ports=z['input_port']
        truth=z['truth'] if ds=='E2' else z['Ez_real']+1j*z['Ez_imag']
    assert len(ids)==len(set(ids))==100 and np.isfinite(truth).all()
    return ep,sm,ids,ports,truth
def historical_rows(ds):
    folder=BASE/('research_e2' if ds=='E2' else 'research_m4');out=[]
    with (folder/'test_per_sample.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if ds=='E2':
                if r['model'] not in ['linear_2400','path_2400']:continue
                method=r['variant'];ck='best'
            else:
                if r['variant'] not in ['linear','path']:continue
                method=r['variant'];ck=r['checkpoint']
            out.append(dict(dataset=ds,method=method,checkpoint=ck,geometry_id=r['geometry_id'],input_port=int(r['input_port']),
                **{k:float(r[k]) for k in ['rel_l2','amplitude_rel_l2','intensity_ratio','intensity_weighted_phase_mae']}))
    return out
@torch.no_grad()
def evaluate():
    verify();assert (OUT/'training_complete.json').exists()
    assert not (OUT/'evaluation_started.json').exists(),'Keep existing evaluation marker'
    save('evaluation_started.json',dict(time=time.time(),training_sha256=sha(OUT/'training_complete.json'),benchmarks=['E2','M4']))
    stage.setup();ff.configure();meta=read(OLD/'manifest.json');rows=[];timings=[];geometry=[];hashes={}
    for ds in ['E2','M4']:
        ep,sm,ids,ports,truth=load_benchmark(ds);x,pc=ff.infer_inputs(ep,sm,meta['calibration'])
        bt=[];lc=[];records=[]
        for e,s,g,p in zip(ep,sm,ids,ports):
            c,r=carrier(e,s,meta['calibration'],g,int(p));bt.append(c);records.append(r)
            f=ff.geometry_features(e,s);k=float(np.polyval(meta['calibration'],float(e.max())))
            lc.append(np.exp(1j*k*f['linear']).astype(np.complex64))
        geometry.extend([dict(dataset=ds,**r) for r in records])
        cs=dict(branch_tangent=torch.from_numpy(np.stack(bt)[:,None]),path=pc,linear=torch.from_numpy(np.stack(lc)[:,None]))
        rows.extend(historical_rows(ds))
        todo=[('branch_tangent','best'),('branch_tangent','last')]
        if ds=='E2':todo+=[('linear','last'),('path','last')]
        for method,ckname in todo:
            root=(OUT if method=='branch_tangent' else OLD)/'runs'/f'{method}_steps2400_seed2026'
            summary=read(root/'summary.json');assert sha(root/(ckname+'.pt'))==summary[ckname+'_sha256']
            ck=torch.load(root/(ckname+'.pt'),map_location='cpu',weights_only=False)
            assert ck['calibration']==meta['calibration'] and ck['scale']==meta['scale']
            m=ff.model_new(2026);m.load_state_dict(ck['state']);m.eval();pred=[];start=time.perf_counter()
            for j in range(0,100,2):
                p=stage.e1.ORIGINAL_FORWARD(m,x[j:j+2].cuda(),cs[method][j:j+2].cuda(),'path','fp32')
                pred.append(p[:,0].cpu().numpy()*meta['scale'])
            pred=np.concatenate(pred);assert np.isfinite(pred).all()
            timings.append(dict(dataset=ds,method=method,checkpoint=ckname,seconds=time.perf_counter()-start))
            path=OUT/'predictions'/f'{ds}_{method}_{ckname}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,prediction=pred,geometry_id=ids,input_port=ports);hashes[path.name]=sha(path)
            metrics=metric_arrays(pred,truth)
            for i,g in enumerate(ids):
                rows.append(dict(dataset=ds,method=method,checkpoint=ckname,geometry_id=str(g),input_port=int(ports[i]),
                    **{k:float(v[i]) for k,v in metrics.items()}))
            print('EVALUATED '+ds+'/'+method+'/'+ckname,flush=True)
            del m,ck,p,pred;torch.cuda.empty_cache()
        del x,pc,cs,ep,sm,truth;torch.cuda.empty_cache()
    assert len(rows)==1200 and len({(r['dataset'],r['method'],r['checkpoint'],r['geometry_id']) for r in rows})==1200
    csvwrite('test_per_geometry.csv',rows);save('inference_costs.json',timings);save('geometry_checks_benchmarks.json',geometry)
    summarize(rows);verify()
    save('preservation_check.json',dict(passed=True,files_checked=len(read(OUT/'freeze.json')['protected_hashes'])))
    save('evaluation_complete.json',dict(seed=2026,datasets=['E2','M4'],rows=1200,new_models=1,prediction_hashes=hashes,
        table_sha256=sha(OUT/'test_per_geometry.csv'),historical_preservation=True))
    print('EVALUATION COMPLETE',flush=True)
def summarize(rows):
    metrics=['rel_l2','amplitude_rel_l2','intensity_ratio','intensity_weighted_phase_mae'];summ=[];paired=[]
    for ds in ['E2','M4']:
        for ck in ['best','last']:
            for group in ['all','port1','port2','port3']:
                data={m:sorted([r for r in rows if r['dataset']==ds and r['checkpoint']==ck and r['method']==m and (group=='all' or r['input_port']==int(group[-1]))],key=lambda r:r['geometry_id']) for m in METHODS}
                n=len(data['linear']);assert n and all(len(v)==n for v in data.values())
                assert all([r['geometry_id'] for r in v]==[r['geometry_id'] for r in data['linear']] for v in data.values())
                draw=np.random.default_rng(2026).integers(0,n,(10000,n))
                for m,rr in data.items():
                    summ.append(dict(dataset=ds,checkpoint=ck,method=m,group=group,n=n,
                        **{k:float(np.mean([r[k] for r in rr])) for k in metrics},
                        rel_l2_median=float(np.median([r['rel_l2'] for r in rr])),rel_l2_p90=float(np.quantile([r['rel_l2'] for r in rr],.9))))
                for a,b in itertools.combinations(METHODS,2):
                    for metric in metrics:
                        av=np.array([r[metric] for r in data[a]]);bv=np.array([r[metric] for r in data[b]]);d=av-bv
                        paired.append(dict(dataset=ds,checkpoint=ck,group=group,baseline=a,candidate=b,metric=metric,n=n,
                            baseline_mean=float(av.mean()),candidate_mean=float(bv.mean()),mean_baseline_minus_candidate=float(d.mean()),
                            relative_reduction=float(1-bv.mean()/av.mean()),positive_differences=int((d>0).sum()),negative_differences=int((d<0).sum()),
                            geometry_bootstrap_ci95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
    for ds in ['E2','M4']:
        for ck in ['best','last']:
            for m in METHODS:
                rr=[r for r in summ if r['dataset']==ds and r['checkpoint']==ck and r['method']==m and r['group'].startswith('port')]
                summ.append(dict(dataset=ds,checkpoint=ck,method=m,group='port_macro',n=100,
                    **{k:float(np.mean([r[k] for r in rr])) for k in metrics}))
    save('test_summary.json',summ);save('paired_comparisons.json',paired)
    curves=[]
    for m in METHODS:
        root=(OUT if m=='branch_tangent' else OLD)/'runs'/f'{m}_steps2400_seed2026'
        curves.extend([dict(method=m,**r) for r in read(root/'history.json')])
    csvwrite('training_curves.csv',curves)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for m in METHODS:
        q=[r for r in curves if r['method']==m];axes[0].plot([r['step'] for r in q],[r['val_rel_l2'] for r in q],label=m)
    axes[0].set(xlabel='Optimizer updates',ylabel='Validation complex RelL2');axes[0].legend(fontsize=8)
    for ax,ds in zip(axes[1:],['E2','M4']):
        values=[next(r['rel_l2'] for r in summ if r['dataset']==ds and r['method']==m and r['checkpoint']=='best' and r['group']=='all') for m in METHODS]
        ax.bar(METHODS,values);ax.tick_params(axis='x',rotation=25);ax.set(title=ds+' seed2026',ylabel='Mean test complex RelL2')
        for i,v in enumerate(values):ax.text(i,v+.008,f'{v:.6f}',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
    lines=['# Y-branch BranchTangent seed2026 pilot','','One new training; historical Linear/Path reused. E2/M4 are previously-used geometry sets, not independent training repetitions.','',
        '| Dataset | Checkpoint | Method | Mean RelL2 | Median | p90 |','|---|---|---|---:|---:|---:|']
    for r in summ:
        if r['group']=='all':lines.append(f"| {r['dataset']} | {r['checkpoint']} | {r['method']} | {r['rel_l2']:.9f} | {r['rel_l2_median']:.9f} | {r['rel_l2_p90']:.9f} |")
    lines+=['','## Path vs BranchTangent','| Dataset | Checkpoint | Group | Mean Path-BT | CI95 geometry | Positive / n |','|---|---|---|---:|---|---:|']
    for r in paired:
        if r['baseline']=='path' and r['candidate']=='branch_tangent' and r['metric']=='rel_l2':
            lines.append(f"| {r['dataset']} | {r['checkpoint']} | {r['group']} | {r['mean_baseline_minus_candidate']:.9f} | {r['geometry_bootstrap_ci95']} | {r['positive_differences']}/{r['n']} |")
    lines+=['','Intervals are conditional on trained models. No training-seed stability, equivalence or causal isolation is established.',
        'Intensity ratio is a field-energy ratio (ideal 1), not a lower-is-better error or certified power.',
        'Hard branch seams are retained and measured in geometry_checks_trainval.json / geometry_checks_benchmarks.json.',
        'No target phase alignment, no extra input channels or new FDTD. All negative differences retained.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train','evaluate']);a=p.parse_args()
    try:
        if a.stage=='preflight':data,records=prepare();preflight(data,records)
        elif a.stage=='train':data,_=prepare();train_bt(data)
        else:evaluate()
    except Exception as exc:
        save('failure_'+a.stage+'.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
