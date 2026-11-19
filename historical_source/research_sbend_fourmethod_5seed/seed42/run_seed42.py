"""Seed42 adapter; historical S-bend sources and all seed2026 outputs remain read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import sys,json,csv,time,itertools,importlib.util,traceback
from pathlib import Path
import numpy as np
BASE=Path('/mnt/d/game');P=Path(__file__).resolve().parent;PREV=P.parent
spec=importlib.util.spec_from_file_location('pilot2026',PREV/'experiment.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
C=pilot.C;model=pilot.model;train=pilot.train;E=pilot.E;torch=pilot.torch;OLD=pilot.OLD
METHODS=pilot.METHODS;SEED=42
read=pilot.read;sha=pilot.sha
def save(n,v):
    p=P/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def csvwrite(n,rows):
    with (P/n).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bind(seed=SEED):
    model.SEED=seed;train.SEED=seed;train.OUT=P
def verify():
    f=read(P/'freeze.json')
    for n,h in f['protected_hashes'].items():assert sha(BASE/n)==h,'Protected changed: '+n
    for n,h in f['source_hashes'].items():assert sha(P/n)==h,'Frozen adapter changed: '+n
def preflight():
    assert not (P/'freeze.json').exists()
    pilot.verify_snapshot();C.verify_freeze()
    previous=read(PREV/'seed2027/freeze.json')
    for n,h in previous['protected_hashes'].items():assert sha(BASE/n)==h
    for n,h in previous['source_hashes'].items():assert sha(PREV/'seed2027'/n)==h
    add=read(PREV/'statistics_addendum_freeze.json');assert sha(PREV/add['file'])==add['sha256']
    model.configure();data,checks=pilot.load_data();replays=[]
    # Seed2026 compatibility: all eight historical checkpoints, same train/validation data.
    bind(2026)
    for name in METHODS:
        root=(PREV if name=='arc_only' else OLD)/'runs'/name
        s=read(root/'summary.json');m=model.new()
        assert model.state_hash(m)==s['initial_state_sha256']
        assert C.ahash(model.batches())==s['batches_sha256']
        for ck in ['best','last']:
            path=root/(ck+'.pt');assert sha(path)==s[ck+'_sha256']
            state=torch.load(path,map_location='cpu',weights_only=False);assert state['seed']==2026
            m.load_state_dict(state['state']);v=train.score(m,data,np.arange(64,80),name)
            expected=s['best_val' if ck=='best' else 'last_val'];err=abs(float(v.mean())-expected)
            assert err<1e-7
            replays.append(dict(method=name,checkpoint=ck,mean_rel_l2=float(v.mean()),error=err))
        del m,state;torch.cuda.empty_cache()
    bind();initials=[];batchhashes=[];rnghashes=[]
    for name in METHODS:
        m=model.new();initials.append(model.state_hash(m));batchhashes.append(C.ahash(model.batches()))
        rnghashes.append(C.ahash(torch.get_rng_state().numpy())+C.ahash(torch.cuda.get_rng_state().cpu().numpy()))
        m.train();out=model.forward(m,data['x'][:2].cuda(),data['c'][name][:2].cuda())
        loss=model.rel_mse(out,data['y'][:2].cuda()).mean();loss.backward()
        assert torch.isfinite(loss) and all(q.grad is None or torch.isfinite(q.grad).all() for q in m.parameters())
        real=sum(q.numel()*(2 if q.is_complex() else 1) for q in m.parameters());assert real==2944802
        del m,out,loss;torch.cuda.empty_cache()
    assert len(set(initials))==len(set(batchhashes))==len(set(rnghashes))==1
    assert initials[0]!=read(PREV/'preflight.json')['model_checks'][0]['initial_state_sha256']
    assert batchhashes[0]!=read(PREV/'preflight.json')['model_checks'][0]['batches_sha256']
    p27=read(PREV/'seed2027/preflight.json')['model_checks'][0]
    assert initials[0]!=p27['initial_state_sha256'] and batchhashes[0]!=p27['batches_sha256']
    save('preflight.json',dict(passed=True,seed=SEED,optimizer_updates=0,test_fields_loaded=False,
        model_checks=[dict(initial_state_sha256=initials[0],batches_sha256=batchhashes[0])],
        common_initial_rng_hash=rnghashes[0],geometry_checks=checks,historical_validation_replays=replays,
        real_scalar_degrees=real,runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))
    protected=dict(read(PREV/'freeze.json')['protected_hashes'])
    # Protect all original pilot artifacts without including this new seed directory.
    for p in PREV.rglob('*'):
        if p.is_file() and P not in p.parents and '__pycache__' not in p.parts:
            protected[str(p.relative_to(BASE))]=sha(p)
    save('freeze.json',dict(time_utc=C.now(),seed=SEED,methods=METHODS,protected_hashes=protected,
        source_hashes={n:sha(P/n) for n in ['run_seed42.py','PROTOCOL.md','THREE_SEED_AMENDMENT.md']},
        preflight_sha256=sha(P/'preflight.json'),no_historical_reuse_for_seed42=True))
    print('PREFLIGHT PASSED; seed42 source frozen; zero optimizer updates',flush=True)
def train_all():
    verify();assert read(P/'preflight.json')['passed'];bind();model.configure();data,_=pilot.load_data()
    result=[]
    for name in METHODS:
        s=train.train_one(data,name);assert s['seed']==SEED and s['steps']==2400
        assert s['batches_sha256']==read(P/'preflight.json')['model_checks'][0]['batches_sha256']
        result.append(s);save('training_progress.json',result);print('METHOD COMPLETE '+name,flush=True)
    assert len({s['initial_state_sha256'] for s in result})==len({s['batches_sha256'] for s in result})==1
    save('training_complete.json',dict(seed=SEED,results=result,time_utc=C.now(),
        weight_hashes={str(p.relative_to(P)):sha(p) for p in (P/'runs').glob('*/*.pt')},test_evaluated=False))
    verify();print('TRAINING COMPLETE ALL FOUR',flush=True)
def summarize(rows):
    metrics=[k for k in rows[0] if k not in ['method','checkpoint','geometry_id','input_port','width_um','R_min_um']]
    summaries=[];pairs=[]
    for ck in ['best','last']:
        for group in ['all','input1','input2']:
            data={m:sorted([r for r in rows if r['method']==m and r['checkpoint']==ck and
                (group=='all' or r['input_port']==int(group[-1]))],key=lambda r:r['geometry_id']) for m in METHODS}
            n=len(data['linear']);assert n and all(len(a)==n for a in data.values())
            assert all([r['geometry_id'] for r in v]==[r['geometry_id'] for r in data['linear']] for v in data.values())
            draw=np.random.default_rng(2026).integers(0,n,(10000,n))
            for m,rr in data.items():
                summaries.append(dict(method=m,checkpoint=ck,group=group,n=n,
                    **{k:float(np.mean([r[k] for r in rr])) for k in metrics},
                    rel_l2_median=float(np.median([r['rel_l2'] for r in rr])),rel_l2_p90=float(np.quantile([r['rel_l2'] for r in rr],.9))))
            for a,b in itertools.combinations(METHODS,2):
                for metric in metrics:
                    av=np.array([r[metric] for r in data[a]]);bv=np.array([r[metric] for r in data[b]]);d=av-bv
                    pairs.append(dict(checkpoint=ck,group=group,baseline=a,candidate=b,metric=metric,n=n,
                        baseline_mean=float(av.mean()),candidate_mean=float(bv.mean()),mean_baseline_minus_candidate=float(d.mean()),
                        relative_error_reduction=float(1-bv.mean()/av.mean()),improved=int((d>0).sum()),worsened=int((d<0).sum()),
                        geometry_bootstrap_ci95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
    save('test_summary.json',summaries);save('paired_comparisons.json',pairs)
    curves=[]
    for m in METHODS:curves.extend([dict(method=m,**r) for r in read(P/'runs'/m/'history.json')])
    csvwrite('training_curves.csv',curves)
    # Three seeds; sequential amendment, no pooled geometry independence claim.
    olds=read(PREV/'test_summary.json');seed27=read(PREV/'seed2027/test_summary.json')
    allsets=[summaries,olds,seed27];seeds=[42,2026,2027];aggregate=[];aggregate_pairs=[]
    for ck in ['best','last']:
        for m in METHODS:
            values=[next(r['rel_l2'] for r in ss if r['method']==m and r['checkpoint']==ck and r['group']=='all') for ss in allsets]
            aggregate.append(dict(method=m,checkpoint=ck,seeds=seeds,seed_mean_rel_l2=values,
                mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),n_training_seeds=3))
        for a,b in itertools.combinations(METHODS,2):
            aa=next(r for r in aggregate if r['method']==a and r['checkpoint']==ck)
            bb=next(r for r in aggregate if r['method']==b and r['checkpoint']==ck)
            delta=np.array(aa['seed_mean_rel_l2'])-np.array(bb['seed_mean_rel_l2'])
            aggregate_pairs.append(dict(checkpoint=ck,baseline=a,candidate=b,seeds=seeds,
                paired_seed_differences=delta.tolist(),mean=float(delta.mean()),sample_sd=float(delta.std(ddof=1)),
                relative_reductions_by_seed=(1-np.array(bb['seed_mean_rel_l2'])/np.array(aa['seed_mean_rel_l2'])).tolist(),
                relative_reduction_of_grand_means=float(1-bb['mean']/aa['mean'])))
    save('three_seed_statistics.json',dict(methods=aggregate,pairs=aggregate_pairs,n_training_seeds=3,
        sequential_seed_selection=True,withdrawn_seed=2025,active_seeds=[42,2026,2027],independent_geometry_observations_not_pooled=True))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,4,figsize=(18,4))
    for m in METHODS:
        q=[r for r in curves if r['method']==m];ax[0].plot([r['step'] for r in q],[r['val_rel_l2'] for r in q],label=m)
    ax[0].set(xlabel='Optimizer updates',ylabel='Validation complex RelL2',yscale='log');ax[0].legend(fontsize=8)
    for axis,ss,seed in zip(ax[1:],[olds,seed27,summaries],[2026,2027,42]):
        v=[next(r['rel_l2'] for r in ss if r['method']==m and r['checkpoint']=='best' and r['group']=='all') for m in METHODS]
        axis.bar(METHODS,v);axis.tick_params(axis='x',rotation=30);axis.set(title=f'seed{seed}',ylabel='Mean test complex RelL2')
        for i,x in enumerate(v):axis.text(i,x+.004,f'{x:.6f}',ha='center',fontsize=8)
    fig.tight_layout();fig.savefig(P/'comparison.png',dpi=180);fig.savefig(P/'comparison.pdf');plt.close(fig)
    lines=['# S-bend seed42 four-method results','',
        'All four models trained fresh for seed42, 2400 updates each; no2025 state reused. Best selected on original16 validation geometries.',
        'Fixed previously-used test benchmark, 100 geometries. No test alignment, tuning, input changes or new FDTD.','',
        '| Method | best test RelL2 | last test RelL2 | best validation | selected step |',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        s=read(P/'runs'/m/'summary.json')
        vals=[next(r['rel_l2'] for r in summaries if r['method']==m and r['checkpoint']==ck and r['group']=='all') for ck in ['best','last']]
        lines.append(f"| {m} | {vals[0]:.9f} | {vals[1]:.9f} | {s['best_val']:.9f} | {s['best_step']} |")
    lines+=['','## Three-seed results, best checkpoints','',
        '| Method | seed42 | seed2026 | seed2027 | three-seed mean | sample SD |','|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        if r['checkpoint']=='best':
            v=r['seed_mean_rel_l2'];lines.append(f"| {r['method']} | {v[0]:.9f} | {v[1]:.9f} | {v[2]:.9f} | {r['mean']:.9f} | {r['sample_sd']:.9f} |")
    lines+=['','## Seed42 paired results: baseline minus candidate','',
        '| Checkpoint | Pair | Mean difference | geometry CI95 | Improved /100 |','|---|---|---:|---|---:|']
    for r in pairs:
        if r['group']=='all' and r['metric']=='rel_l2':
            lines.append(f"| {r['checkpoint']} | {r['baseline']} to {r['candidate']} | {r['mean_baseline_minus_candidate']:.9f} | {r['geometry_bootstrap_ci95']} | {r['improved']} |")
    lines+=['','Geometry intervals are conditional on fixed models, exploratory, no multiplicity adjustment.',
        'Active seeds42/2026/2027 selected after earlier results and partial2025 validation. User cancelled2025 before test evaluation;2025 is excluded. No formal equivalence/broad stability claim. Seeds2028-2030 not executed.',
        'Current-batch training MSE is not whole-train RelL2. Port proxies are not certified native S/power.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
@torch.no_grad()
def evaluate():
    verify();trained=read(P/'training_complete.json');bind();model.configure()
    for n,h in trained['weight_hashes'].items():assert sha(P/n)==h
    assert not (P/'evaluation_started.json').exists(),'Preserve one-evaluation marker'
    save('evaluation_started.json',dict(time_utc=C.now(),seed=SEED,training_sha256=sha(P/'training_complete.json'),
        policy='retrospective fixed benchmark; frozen best/last'))
    records=[r for r in read(OLD/'geometry_manifest.json')['records'] if r['split']=='test'];assert len(records)==100
    cal=read(OLD/'calibration.json');proxy=E.proxy_function()
    xx=[];cc={m:[] for m in METHODS};truth=[];regions=[];args=[];targets=[];native=[]
    for r in records:
        c,_=pilot.arc(r);cc['arc_only'].append(c)
        with np.load(OLD/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
            xx.append(z['x']);regions.append({k[7:]:z[k] for k in z.files if k.startswith('region_')})
            for m in METHODS:
                if m!='arc_only':cc[m].append(z['carrier_'+m])
        with np.load(OLD/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:
            t=z['Ez'].astype(np.complex128);truth.append(t);a=(z['eps'],z['port_masks'],z['port_ids'],r['input_port'],1.55,.05)
            args.append(a);targets.append(proxy(t,*a));native.append(z['S_native'])
    xx=torch.from_numpy(np.stack(xx));cc={m:torch.from_numpy(np.stack(a)) for m,a in cc.items()}
    rows=[];coeff=[];timings=[];hashes={};(P/'predictions').mkdir(exist_ok=True)
    for name in METHODS:
        for ckname in ['best','last']:
            path=P/'runs'/name/(ckname+'.pt');s=read(path.parent/'summary.json');assert sha(path)==s[ckname+'_sha256']
            ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['seed']==SEED and ck['config']==model.CONFIG
            m=model.new();m.load_state_dict(ck['state']);m.eval();pred=[];start=time.perf_counter()
            for j in range(0,100,2):
                p=model.forward(m,xx[j:j+2].cuda(),cc[name][j:j+2].cuda())
                pred.extend((p[:,0]*cal['scale']).cpu().numpy())
            pred=np.stack(pred);assert np.isfinite(pred).all();timings.append(dict(method=name,checkpoint=ckname,seconds=time.perf_counter()-start))
            out=P/'predictions'/f'{name}_{ckname}.npz'
            np.savez_compressed(out,prediction=pred,geometry_id=np.array([r['geometry_id'] for r in records]),
                input_port=np.array([r['input_port'] for r in records]));hashes[out.name]=sha(out)
            for i,r in enumerate(records):
                c=proxy(pred[i].astype(np.complex128),*args[i]);assert np.isfinite(c).all();o=1 if r['input_port']==1 else 0
                rows.append(dict(method=name,checkpoint=ckname,geometry_id=r['geometry_id'],input_port=r['input_port'],
                    width_um=r['params']['wg_width_um'],R_min_um=r['params']['R_min_um'],**E.field_metrics(pred[i],truth[i],regions[i]),
                    **{'noninput_'+k:v for k,v in E.coefficient_errors(c[o],targets[i][o]).items()}))
                for ref,t in [('truth_proxy',targets[i]),('Meep_native',native[i])]:
                    for o in range(2):
                        coeff.append(dict(method=name,checkpoint=ckname,geometry_id=r['geometry_id'],input_port=r['input_port'],
                            output_port=o+1,is_reflection=o+1==r['input_port'],reference=ref,
                            estimate_real=float(c[o].real),estimate_imag=float(c[o].imag),target_real=float(t[o].real),target_imag=float(t[o].imag),
                            **E.coefficient_errors(c[o],t[o])))
            print('EVALUATED '+name+'/'+ckname,flush=True);del m,ck,p,pred;torch.cuda.empty_cache()
    assert len(rows)==800 and len({(r['method'],r['checkpoint'],r['geometry_id']) for r in rows})==800
    csvwrite('test_per_geometry.csv',rows);csvwrite('port_coefficients.csv',coeff);save('inference_costs.json',timings)
    summarize(rows);verify()
    save('preservation_check.json',dict(passed=True,files_checked=len(read(P/'freeze.json')['protected_hashes'])))
    save('evaluation_complete.json',dict(seed=SEED,prediction_hashes=hashes,rows=800,
        table_sha256=sha(P/'test_per_geometry.csv'),historical_files_unchanged=True))
    print('EVALUATION COMPLETE',flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['preflight','train','evaluate']);a=ap.parse_args()
    try:
        if a.stage=='preflight':preflight()
        elif a.stage=='train':train_all()
        else:evaluate()
    except Exception as e:
        save('failure_'+a.stage+'.json',dict(error=repr(e),traceback=traceback.format_exc(),time_utc=C.now()));raise
