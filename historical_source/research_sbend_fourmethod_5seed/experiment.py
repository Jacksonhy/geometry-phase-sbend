"""Isolated ArcOnly extension. Historical inputs are read-only; seed2026 only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,csv,hashlib,time,platform,itertools,argparse,traceback
from pathlib import Path
import numpy as np
BASE=Path('/mnt/d/game')
OLD=BASE/'research_sbend_formal'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OLD))
import common as C
import model,train
import evaluate as E
import torch
METHODS=['linear','arc_only','local_tangent','path_v2']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):return C.sha(p)
def csvwrite(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protected():
    paths=set((BASE/n) for n in read(OLD/'code_freeze.json')['sources'])
    for n in ['code_freeze.json','geometry_manifest.json','geometry_freeze.json','calibration.json',
              'cache_complete.json','data_complete.json','config.json','preflight.json',
              'training_complete.json','evaluation_complete.json','evaluation_started.json',
              'test_per_geometry.csv','test_summary.json','paired_comparisons.json','port_coefficients.csv']:
        paths.add(OLD/n)
    for m in ['linear','local_tangent','path_v2']:
        paths.update((OLD/'runs'/m).glob('*'))
        for ck in ['best','last']:paths.add(OLD/'predictions'/f'{m}_{ck}.npz')
    for r in read(OLD/'cache_complete.json')['records']:paths.add(OLD/'cache'/(r['geometry_id']+'.npz'))
    for r in read(OLD/'data_complete.json')['records']:paths.add(OLD/'data'/r['geometry_id']/'sample.npz')
    return {str(p.relative_to(BASE)):sha(p) for p in sorted(paths) if p.is_file()}
def arc(r):
    # Only field-free keys are loaded; Ez is not accessed here.
    with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
        x=z['x_um'];y=z['y_um'];src=z['source_center_um'];epmax=float(z['eps'].max())
    dx=float(np.median(np.diff(x)));poly,_=C.v1.centerline_from_params(r['params'],(float(x[0]-dx/2),float(x[-1]+dx/2)))
    _,_,s,_=C.v1.curve_info(poly)
    ss=float(C.v1.project_points(np.asarray(src[:2])[None],poly)['s'][0])
    sc=np.interp(x,poly[:,0],s);sig=1 if r['input_port']==1 else -1
    d=np.broadcast_to(sig*(sc-ss),(len(y),len(x))).copy()
    beta=float(np.polyval(read(OLD/'calibration.json')['coefficients_rad_um'],epmax))
    carrier=np.exp(1j*beta*d).astype(np.complex64)[None]
    # Reconstruct historical LocalTangent independently as a compatibility check.
    yc=np.interp(x,poly[:,0],poly[:,1]);ty=C.v1.smooth_tangent_at_x(x,poly)[:,1]
    lt=sig*(sc[None,:]+ty[None,:]*(y[:,None]-yc[None,:])-ss)
    clt=np.exp(1j*beta*lt).astype(np.complex64)[None]
    with np.load(OLD/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
        assert np.array_equal(clt,z['carrier_local_tangent']),'LocalTangent reconstruction mismatch'
        assert beta==float(z['beta_rad_um'])
        linear=z['carrier_linear']
    straight=(x<poly[1,0])|(x>poly[-2,0])
    # Arc and linear have a constant phase difference on either straight lead.
    diff=d[0]-sig*(x-src[0])
    lead_res=max(float(np.ptp(diff[x<poly[1,0]])),float(np.ptp(diff[x>poly[-2,0]])))
    assert lead_res<1e-12
    anchor=float(sig*(np.interp(src[0],poly[:,0],s)-ss));assert abs(anchor)<1e-10
    assert np.max(abs(abs(carrier)-1))<2e-7 and np.all(np.isfinite(carrier))
    assert np.all(np.diff(d[0])*sig>0)
    # Independent scalar polyline integration oracle at three columns.
    for j in [0,len(x)//2,len(x)-1]:
        k=int(np.clip(np.searchsorted(poly[:,0],x[j],side='right')-1,0,len(poly)-2))
        oracle=sum(np.linalg.norm(poly[i+1]-poly[i]) for i in range(k))
        oracle+=np.linalg.norm(poly[k+1]-poly[k])*(x[j]-poly[k,0])/(poly[k+1,0]-poly[k,0])
        assert abs(d[0,j]-sig*(oracle-ss))<1e-12
    return carrier,dict(geometry_id=r['geometry_id'],input_port=r['input_port'],anchor_um=anchor,
                        straight_lead_difference_range_um=lead_res,local_tangent_exact=True)
def load_data():
    records=[r for r in read(OLD/'geometry_manifest.json')['records'] if r['split'] in ['train','val']]
    records.sort(key=lambda r:0 if r['split']=='train' else 1)
    assert len(records)==80 and all(r['split']=='train' for r in records[:64])
    xx=[];yy=[];cc={m:[] for m in METHODS};checks=[]
    cal=read(OLD/'calibration.json')
    for r in records:
        with np.load(OLD/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
            xx.append(z['x'])
            for m in METHODS:
                if m!='arc_only':cc[m].append(z['carrier_'+m])
        with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
            assert str(z['split']) in ['train','val'];yy.append((z['Ez'][None]/cal['scale']).astype(np.complex64))
        c,ch=arc(r);cc['arc_only'].append(c);checks.append(ch)
    return dict(x=torch.from_numpy(np.stack(xx)),y=torch.from_numpy(np.stack(yy)),
                c={m:torch.from_numpy(np.stack(a)) for m,a in cc.items()},records=records,meta=cal),checks
def verify_snapshot():
    for n,h in read(OUT/'freeze.json')['protected_hashes'].items():
        assert sha(BASE/n)==h,'Protected input changed: '+n
    for n,h in read(OUT/'freeze.json')['new_source_hashes'].items():
        assert sha(OUT/n)==h,'New source changed: '+n
def preflight(data,checks):
    assert not (OUT/'freeze.json').exists(),'Already frozen; use train/evaluate stages'
    C.verify_freeze()
    cal=data['meta'];assert cal['train_only'] and not cal['validation_or_test_used'] and cal['passed']
    assert len(cal['rows'])==64
    for r in read(OLD/'cache_complete.json')['records']:
        assert sha(OLD/'cache'/(r['geometry_id']+'.npz'))==r['cache_sha256']
    for r in read(OLD/'data_complete.json')['records']:
        assert sha(OLD/'data'/r['geometry_id']/'sample.npz')==r['sample_sha256']
    evalrec=read(OLD/'evaluation_complete.json')
    for p,h in evalrec['metric_files_sha256'].items():assert sha(OLD/p)==h
    replay=[]
    model.configure()
    for name in ['linear','local_tangent','path_v2']:
        s=read(OLD/'runs'/name/'summary.json')
        assert s['seed']==2026 and s['steps']==2400
        m=model.new();initial=model.state_hash(m);bh=C.ahash(model.batches())
        assert initial==s['initial_state_sha256'] and bh==s['batches_sha256']
        for ck in ['best','last']:
            path=OLD/'runs'/name/(ck+'.pt');assert sha(path)==s[ck+'_sha256']
            c=torch.load(path,map_location='cpu',weights_only=False)
            assert c['initial_state_sha256']==initial and c['batches_sha256']==bh
            assert c['config']==model.CONFIG and c['seed']==2026
            m.load_state_dict(c['state']);values=train.score(m,data,np.arange(64,80),name)
            expected=s['best_val' if ck=='best' else 'last_val']
            err=abs(float(values.mean())-expected);assert err<1e-7,(name,ck,err)
            replay.append(dict(method=name,checkpoint=ck,replay_mean=float(values.mean()),historical_mean=expected,error=err))
        del m,c;torch.cuda.empty_cache()
    m=model.new();initial=model.state_hash(m);bh=C.ahash(model.batches())
    assert initial==read(OLD/'preflight.json')['model_checks'][0]['initial_state_sha256']
    m.train();p=model.forward(m,data['x'][:2].cuda(),data['c']['arc_only'][:2].cuda())
    loss=model.rel_mse(p,data['y'][:2].cuda()).mean();loss.backward()
    assert torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in m.parameters())
    real=sum(q.numel()*(2 if q.is_complex() else 1) for q in m.parameters());assert real==2944802
    del m,p,loss;torch.cuda.empty_cache()
    runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
    save('preflight.json',dict(passed=True,optimizer_updates=0,test_fields_loaded=False,geometry_checks=checks,
         historical_validation_replays=replay,runtime=runtime,real_scalar_degrees=real,
         model_checks=[dict(initial_state_sha256=initial,batches_sha256=bh)]))
    snapshot=protected()
    save('freeze.json',dict(time_utc=C.now(),planned_seeds=[2026,2027,2028,2029,2030],active_seeds=[2026],
         methods=METHODS,protected_hashes=snapshot,new_source_hashes={n:sha(OUT/n) for n in ['experiment.py','PROTOCOL.md']},
         preflight_sha256=sha(OUT/'preflight.json'),historical_reuse_methods=['linear','local_tangent','path_v2'],
         no_new_input=True,no_test_based_tuning=True))
    print('PREFLIGHT PASSED; historical validation replay exact within 1e-7; protocol frozen',flush=True)
def train_arc(data):
    verify_snapshot();assert read(OUT/'preflight.json')['passed']
    train.OUT=OUT;train.SEED=2026
    model.configure()
    result=train.train_one(data,'arc_only')
    assert result['initial_state_sha256']==read(OUT/'preflight.json')['model_checks'][0]['initial_state_sha256']
    save('training_complete.json',dict(time_utc=C.now(),seed=2026,arc_only=result,
         historical_reuse={m:read(OLD/'runs'/m/'summary.json') for m in ['linear','local_tangent','path_v2']},
         freeze_sha256=sha(OUT/'freeze.json')))
    print('TRAINING COMPLETE',flush=True)
@torch.no_grad()
def evaluate():
    verify_snapshot();assert (OUT/'training_complete.json').exists()
    assert not (OUT/'evaluation_started.json').exists(),'Evaluation already started; preserve marker'
    save('evaluation_started.json',dict(time_utc=C.now(),training_complete_sha256=sha(OUT/'training_complete.json'),
        test_policy='retrospective fixed benchmark; best/last frozen'))
    records=[r for r in read(OLD/'geometry_manifest.json')['records'] if r['split']=='test'];assert len(records)==100
    cal=read(OLD/'calibration.json');proxy=E.proxy_function()
    xx=[];cc=[];truth=[];regions=[];args=[];targets=[];native=[]
    for r in records:
        c,_=arc(r);cc.append(c)
        with np.load(OLD/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
            xx.append(z['x']);regions.append({k[7:]:z[k] for k in z.files if k.startswith('region_')})
        with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
            t=z['Ez'].astype(np.complex128);truth.append(t)
            a=(z['eps'],z['port_masks'],z['port_ids'],r['input_port'],1.55,.05)
            args.append(a);targets.append(proxy(t,*a));native.append(z['S_native'])
    xx=torch.from_numpy(np.stack(xx));cc=torch.from_numpy(np.stack(cc))
    with (OLD/'test_per_geometry.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[]
        for row in csv.DictReader(f):
            if row['method'] not in METHODS:continue
            rows.append({k:(v if k in ['method','checkpoint','geometry_id'] else int(v) if k=='input_port' else float(v)) for k,v in row.items()})
    coeff=[]
    (OUT/'predictions').mkdir(exist_ok=True);model.configure();timings=[]
    for checkpoint in ['best','last']:
        path=OUT/'runs/arc_only'/(checkpoint+'.pt')
        expected=read(OUT/'training_complete.json')['arc_only'][checkpoint+'_sha256'];assert sha(path)==expected
        m=model.new();ck=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(ck['state']);m.eval()
        predictions=[];start=time.perf_counter()
        for j in range(0,100,2):
            p=model.forward(m,xx[j:j+2].cuda(),cc[j:j+2].cuda())
            predictions.extend((p[:,0]*cal['scale']).cpu().numpy())
        pred=np.stack(predictions);assert np.isfinite(pred).all()
        timings.append(dict(checkpoint=checkpoint,seconds=time.perf_counter()-start))
        np.savez_compressed(OUT/'predictions'/f'arc_only_{checkpoint}.npz',prediction=pred,
             geometry_id=np.array([r['geometry_id'] for r in records]),input_port=np.array([r['input_port'] for r in records]))
        for i,r in enumerate(records):
            c=proxy(pred[i].astype(np.complex128),*args[i]);assert np.isfinite(c).all()
            o=1 if r['input_port']==1 else 0
            row=dict(method='arc_only',checkpoint=checkpoint,geometry_id=r['geometry_id'],input_port=r['input_port'],
                width_um=r['params']['wg_width_um'],R_min_um=r['params']['R_min_um'],**E.field_metrics(pred[i],truth[i],regions[i]),
                **{'noninput_'+k:v for k,v in E.coefficient_errors(c[o],targets[i][o]).items()})
            rows.append(row)
            for ref,t in [('truth_proxy',targets[i]),('Meep_native',native[i])]:
                for o in range(2):
                    coeff.append(dict(method='arc_only',checkpoint=checkpoint,geometry_id=r['geometry_id'],input_port=r['input_port'],
                        output_port=o+1,is_reflection=o+1==r['input_port'],reference=ref,
                        estimate_real=float(c[o].real),estimate_imag=float(c[o].imag),
                        target_real=float(t[o].real),target_imag=float(t[o].imag),**E.coefficient_errors(c[o],t[o])))
        del m,ck,p,pred;torch.cuda.empty_cache()
    assert len(rows)==800
    csvwrite('test_per_geometry.csv',rows);csvwrite('arc_only_port_coefficients.csv',coeff)
    save('inference_costs.json',timings)
    summarize(rows)
    verify_snapshot()
    save('preservation_check.json',dict(passed=True,files_checked=len(read(OUT/'freeze.json')['protected_hashes']),time_utc=C.now()))
    save('evaluation_complete.json',dict(time_utc=C.now(),seed=2026,methods=METHODS,geometries=100,
        new_models_evaluated=2,historical_methods_reused=3,prediction_sha256={p.name:sha(p) for p in (OUT/'predictions').glob('*.npz')},
        table_sha256=sha(OUT/'test_per_geometry.csv'),preservation_passed=True))
    print('EVALUATION COMPLETE',flush=True)
def summarize(rows):
    metrics=[k for k in rows[0] if k not in ['method','checkpoint','geometry_id','input_port','width_um','R_min_um']]
    summaries=[];pairs=[]
    for ck in ['best','last']:
        for port in [0,1,2]:
            data={m:sorted([r for r in rows if r['checkpoint']==ck and r['method']==m and (port==0 or r['input_port']==port)],key=lambda r:r['geometry_id']) for m in METHODS}
            n=len(data['linear']);assert all(len(v)==n for v in data.values())
            assert all([r['geometry_id'] for r in v]==[r['geometry_id'] for r in data['linear']] for v in data.values())
            draw=np.random.default_rng(2026).integers(0,n,(10000,n))
            for m,rr in data.items():
                summaries.append(dict(method=m,checkpoint=ck,group='all' if port==0 else f'input{port}',n=n,
                    **{k:float(np.mean([r[k] for r in rr])) for k in metrics},
                    rel_l2_median=float(np.median([r['rel_l2'] for r in rr])),rel_l2_p90=float(np.quantile([r['rel_l2'] for r in rr],.9))))
            for a,b in itertools.combinations(METHODS,2):
                for metric in metrics:
                    av=np.array([r[metric] for r in data[a]]);bv=np.array([r[metric] for r in data[b]]);d=av-bv
                    pairs.append(dict(checkpoint=ck,group='all' if port==0 else f'input{port}',baseline=a,candidate=b,metric=metric,
                        n=n,baseline_mean=float(av.mean()),candidate_mean=float(bv.mean()),mean_baseline_minus_candidate=float(d.mean()),
                        relative_error_reduction=float(1-bv.mean()/av.mean()),improved=int((d>0).sum()),worsened=int((d<0).sum()),
                        geometry_bootstrap_ci95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
    save('test_summary.json',summaries);save('paired_comparisons.json',pairs)
    curves=[]
    for m in METHODS:
        path=(OUT if m=='arc_only' else OLD)/'runs'/m/'history.json'
        curves.extend([dict(method=m,**r) for r in read(path)])
    csvwrite('training_curves.csv',curves)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for m in METHODS:
        q=[r for r in curves if r['method']==m];ax[0].plot([r['step'] for r in q],[r['val_rel_l2'] for r in q],label=m)
    ax[0].set(xlabel='Optimizer updates',ylabel='Validation complex RelL2',yscale='log');ax[0].legend(fontsize=8)
    best=[next(r for r in summaries if r['checkpoint']=='best' and r['group']=='all' and r['method']==m)['rel_l2'] for m in METHODS]
    ax[1].bar(METHODS,best);ax[1].tick_params(axis='x',rotation=20);ax[1].set(ylabel='Mean test complex RelL2',title='seed2026; validation-selected checkpoints')
    for i,v in enumerate(best):ax[1].text(i,v+.005,f'{v:.6f}',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'seed2026_comparison.png',dpi=180);fig.savefig(OUT/'seed2026_comparison.pdf');plt.close(fig)
    a=dict(zip(METHODS,best))
    report=['# Seed2026 four-reference pilot','',
        'Single training seed, already-used fixed test benchmark. Historical Linear/LocalTangent/v2 are reused after checks.',
        'No cross-seed stability or formal equivalence claim is established.','',
        '| Method | best test RelL2 | last test RelL2 | best validation | selected step |',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        summ=read((OUT if m=='arc_only' else OLD)/'runs'/m/'summary.json')
        last=next(r['rel_l2'] for r in summaries if r['method']==m and r['checkpoint']=='last' and r['group']=='all')
        report.append(f"| {m} | {a[m]:.9f} | {last:.9f} | {summ['best_val']:.9f} | {summ['best_step']} |")
    report+=['','## Paired field results (baseline minus candidate; positive favors candidate)',
        '| Checkpoint | Pair | Mean difference | geometry CI95 | Improved / 100 |','|---|---|---:|---|---:|']
    for r in pairs:
        if r['group']=='all' and r['metric']=='rel_l2':
            report.append(f"| {r['checkpoint']} | {r['baseline']} vs {r['candidate']} | {r['mean_baseline_minus_candidate']:.9f} | {r['geometry_bootstrap_ci95']} | {r['improved']} |")
    report+=['',f"ArcOnly versus Linear mean error reduction: {100*(1-a['arc_only']/a['linear']):.3f}%.",
        f"LocalTangent versus ArcOnly mean error reduction: {100*(1-a['local_tangent']/a['arc_only']):.3f}%.",
        f"ArcOnly captures {100*(a['linear']-a['arc_only'])/(a['linear']-a['path_v2']):.3f}% of the observed Linear-to-v2 mean-error reduction.",
        '', 'Interpretation must follow measured ArcOnly ordering; these percentages are descriptive for this seed.',
        'All six pairs, directions, regional and proxy metrics are in JSON/CSV. No test phase alignment.',
        'Only seed2026 executed. Seeds2027–2030 remain planned.']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','train','evaluate']);a=parser.parse_args()
    try:
        if a.stage=='preflight':
            model.configure();data,checks=load_data();preflight(data,checks)
        elif a.stage=='train':
            model.configure();data,_=load_data();train_arc(data)
        else:evaluate()
    except Exception as exc:
        save('failure_'+a.stage+'.json',dict(error=repr(exc),traceback=traceback.format_exc(),time_utc=C.now()));raise
