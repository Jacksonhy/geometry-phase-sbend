"""Frozen-weight Y-branch regional diagnostics; NO TRAINING entry points."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MPLBACKEND']='Agg'
import sys,json,csv,time,hashlib,argparse,importlib.util
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt,binary_dilation,binary_erosion
from scipy.stats import spearmanr,t as student_t
B=Path('/mnt/d/game');O=Path(__file__).resolve().parent;P=B/'research_ybranch_branch_tangent_seed2026'
sp=importlib.util.spec_from_file_location('pilot',P/'pilot.py');pilot=importlib.util.module_from_spec(sp);sp.loader.exec_module(pilot)
ff=pilot.ff
REG=['junction','boundary','branch_core','trunk_core','near_cladding','far_cladding']
PARAM=['wg_width_um','l_junction_um','l_bend_um','h_bend_um','l_out_um']
RULE=dict(pixel_um=.05,boundary_um=.2,junction_x_half_um=1.,junction_y_half_um=1.25,near_cladding_um=.5,
          boundary_sensitivity_um=[.1,.2,.4],junction_sensitivity_um=[.5,1.,1.5],
          intensity_threshold=.01,intensity_sensitivity=[.001,.01,.05],smooth_buffer_pixels=2,exclude_outer_pixels=2,
          bootstrap=2000,seed=13011)
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(n,x):
    p=O/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csvout(n,rows):
    if not rows:return
    p=O/n;p.parent.mkdir(exist_ok=True,parents=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p):return pilot.sha(p)
def log(x):print(x,flush=True)
def load(ds):
    if ds=='DEV':
        with np.load(B/'picflow_ybranch_smoke_80.npz') as z:
            ix=np.flatnonzero(z['split']!='test')
            d={k:z[k][ix] for k in ['eps','src_mask','geometry_id','input_port','split','source_shard','source_slot']}
            d['truth']=(z['Ez_real'][ix]+1j*z['Ez_imag'][ix])
    else:
        ep,sm,ids,ports,truth=pilot.load_benchmark(ds)
        d=dict(eps=ep,src_mask=sm,geometry_id=ids,input_port=ports,truth=truth,split=np.array([ds]*len(ids)))
    return d
def parameters(ds,d):
    if ds=='DEV':records=[dict(shard=str(sh),slot=int(sl)) for sh,sl in zip(d['source_shard'],d['source_slot'])]
    else:
        name='research_e2/manifest.json' if ds=='E2' else 'research_m4/materials_manifest.json'
        index={r['geometry_id']:r for r in read(B/name)['selection']}
        records=[index[str(g)] for g in d['geometry_id']]
    result=[]
    for r in records:
        if 'params' in r:q={k:float(r['params'][k]) for k in PARAM}
        else:
            sh=r.get('shard',r.get('source_shard'));slot=r.get('slot',r.get('source_slot'))
            with np.load(B/'picflow_source/shards'/sh) as z:
                q={k:float(z[f's{slot}/params/{k}']) for k in PARAM}
        result.append(q)
    return result
def prediction(ds,m,ck,ids):
    if ds=='DEV':p=O/'predictions'/f'DEV_{m}_{ck}.npz'
    elif m=='branch_tangent':p=P/'predictions'/f'{ds}_{m}_{ck}.npz'
    elif ds=='E2':p=(B/'research_e2/predictions/path_2400.npz') if ck=='best' else P/'predictions/E2_path_last.npz'
    else:p=B/'research_m4/predictions'/f'path_{ck}.npz'
    with np.load(p) as z:
        assert np.array_equal(z['geometry_id'],ids)
        return z['prediction']
def geometry(ep,src):
    f=ff.geometry_features(ep,src);u,info=pilot.correction(f)
    k=float(np.polyval(read(B/'research_e3_2400/manifest.json')['calibration'],float(ep.max())))
    distances=[f['path'].astype(float),f['path'].astype(float)+u]
    carriers=[np.exp(1j*k*f['path']).astype(np.complex64),np.exp(1j*k*distances[1]).astype(np.complex64)]
    h,w=ep.shape;yy,xx=np.mgrid[:h,:w]
    # Every adjacent grid edge, including routing sign flips. Trunk label seam with identical paths excluded geometrically.
    edges={}
    for kind in ['branch','route','union']:
        ls=[]
        for ax in [0,1]:
            a=[slice(None)]*2;b=[slice(None)]*2;a[ax]=slice(None,-1);b[ax]=slice(1,None);a=tuple(a);b=tuple(b)
            branch=(info['branch'][a]!=info['branch'][b]) & (np.maximum(xx[a],xx[b])>f['junction'])
            route=info['q'][a]!=info['q'][b]
            ls.append(branch if kind=='branch' else route if kind=='route' else branch|route)
        edges[kind]=ls
    pix=np.zeros((h,w),bool)
    for ax,e in enumerate(edges['union']):
        a=[slice(None)]*2;b=[slice(None)]*2;a[ax]=slice(None,-1);b[ax]=slice(1,None)
        pix[tuple(a)]|=e;pix[tuple(b)]|=e
    f.update(u=u,info=info,k=k,distances=distances,carriers=carriers,edge_pairs=edges,seam=pix,xx=xx,yy=yy)
    return f
def partition(f,bw=.2,jx=1.):
    xx=f['xx'];yy=f['yy'];j=f['junction'];um=RULE['pixel_um']
    mid=(f['upper'][j]+f['lower'][j])/2
    junction=(abs(xx-j)*um<=jx)&(abs(yy-mid)*um<=RULE['junction_y_half_um'])
    band=(distance_transform_edt(~f['seam'])*um<=bw)&~junction
    branch=f['core']&(xx>j)&~junction&~band
    trunk=f['core']&~junction&~band&~branch
    near=(distance_transform_edt(~f['core'])*um<=RULE['near_cladding_um'])&~junction&~band&~branch&~trunk
    far=~(junction|band|branch|trunk|near)
    masks=[junction,band,branch,trunk,near,far]
    assert np.all(np.stack(masks).sum(0)==1)
    return masks
def edgevalues(a,ax):
    return (a[:-1],a[1:]) if ax==0 else (a[:,:-1],a[:,1:])
def burden(E,f):
    I=abs(E.astype(np.complex128))**2;den=I.sum();out={}
    for kind,edges in f['edge_pairs'].items():
        wsum=0.;nums=np.zeros(2);anums=np.zeros(2);allw=0.
        for ax,edge in enumerate(edges):
            ia,ib=edgevalues(I,ax);w=(ia+ib)/2;allw+=w.sum();wsum+=w[edge].sum()
            for m,c in enumerate(f['carriers']):
                ca,cb=edgevalues(c,ax);nums[m]+=(w*abs(cb-ca)**2)[edge].sum()
                aa,ab=edgevalues(E*np.conj(c),ax);anums[m]+=(abs(ab-aa)**2)[edge].sum()
        out[kind+'_edge_energy_fraction']=float(wsum/allw)
        for m,name in enumerate(['path','bt']):
            out[kind+'_J_'+name]=float(nums[m]/wsum) if wsum else 0.
            out[kind+'_B_'+name]=float(nums[m]/den)
            out[kind+'_A_jump_'+name]=float(anums[m]/den)
        for name in ['J','B','A_jump']:out[kind+'_delta_'+name]=out[kind+'_'+name+'_bt']-out[kind+'_'+name+'_path']
    out['weighted_correction_phase_rms']=float(np.sqrt((I*(f['k']*f['u'])**2).sum()/den))
    return out
def phase(E,f,masks,threshold=.01,maps=False):
    E=E.astype(np.complex128);I=abs(E)**2;peak=I[f['core']].max()
    # Exclude artificial seams, material jumps and outer 2 pixels for centered derivatives.
    smooth=~binary_dilation(f['seam'],iterations=RULE['smooth_buffer_pixels'])
    material=f['core']^binary_erosion(f['core'])
    smooth &= ~binary_dilation(material,iterations=2)
    smooth[:2]=False;smooth[-2:]=False;smooth[:,:2]=False;smooth[:,-2:]=False
    smooth&=I>=threshold*peak
    ey,ex=np.gradient(E);ke=[np.imag(np.conj(E)*ey)/(I+1e-30),np.imag(np.conj(E)*ex)/(I+1e-30)]
    ds=[np.gradient(d) for d in f['distances']]
    rows=[];mapv=[]
    for mm in range(2):mapv.append(sum((ke[a]-f['k']*ds[mm][a])**2 for a in [0,1]))
    for name,mask in [('all',np.ones_like(smooth)),*zip(REG,masks)]:
        good=smooth&mask;weight=I[good].sum()
        r=dict(region=name,smooth_energy_fraction=float(weight/I.sum()),smooth_pixels=int(good.sum()))
        for mm,label in enumerate(['path','bt']):
            r['D_'+label]=float((I*mapv[mm])[good].sum()/weight) if weight else None
            num=0.;den=0.;wrapping=0.
            for ax in [0,1]:
                ga,gb=edgevalues(good,ax);g=ga&gb
                ea,eb=edgevalues(E,ax);ia,ib=edgevalues(I,ax);w=(ia+ib)/2
                ca,cb=edgevalues(f['carriers'][mm],ax)
                residual=np.angle(eb*np.conj(ea)*np.conj(cb*np.conj(ca)))
                num+=(w*residual**2)[g].sum();den+=w[g].sum()
                wrapping+=(w*(abs(residual)>.9*np.pi))[g].sum()
            r['Q_'+label]=float(num/den) if den else None
            r['near_pi_fraction_'+label]=float(wrapping/den) if den else None
        r['delta_D']=r['D_bt']-r['D_path'] if weight else None
        r['delta_Q']=r['Q_bt']-r['Q_path'] if r['Q_bt'] is not None else None
        rows.append(r)
    return (rows,mapv,smooth) if maps else rows
def fit_controls(rows):
    results=[];associations=[]
    for ds in sorted(set(r['dataset'] for r in rows)):
      for ck in ['best','last']:
        rr=[r for r in rows if r['dataset']==ds and r['checkpoint']==ck];n=len(rr)
        if n<25:continue
        cov=np.array([[r[k] for k in PARAM+['epsmax','e_path']] for r in rr]);sd=cov.std(0);cov=(cov-cov.mean(0))/np.where(sd>1e-10,sd,1)
        ports=np.array([r['input_port'] for r in rr])
        X=np.column_stack([np.ones(n),ports==2,ports==3,cov])
        rank=np.linalg.matrix_rank(X)
        for response in ['delta_e','relative_delta','delta_e2']:
            y=np.array([r[response] for r in rr]);pinv=np.linalg.pinv(X);coef=pinv@y;res=y-X@coef
            leverage=np.einsum('ij,ji->i',X,pinv)
            hc=(pinv*(res/np.maximum(1-leverage,1e-6))**2)@pinv.T
            crit=student_t.ppf(.975,n-rank)
            for j in [1,2]:
                se=np.sqrt(max(0,hc[j,j]));results.append(dict(dataset=ds,checkpoint=ck,response=response,contrast=f'port{j+1}-port1',adjusted_difference=float(coef[j]),
                    ci_low=float(coef[j]-crit*se),ci_high=float(coef[j]+crit*se),n=n,design_rank=int(rank),design_columns=X.shape[1],
                    baseline_covariate='e_path; actual five geometry parameters; epsmax',interpretation='observational OLS HC3; conditional on frozen models'))
        y=np.array([r['delta_e'] for r in rr]);yr=y-X@(np.linalg.pinv(X)@y)
        for metric in ['union_delta_B','union_delta_J','phase_delta_Q','phase_delta_D','weighted_correction_phase_rms','slope_roughness']:
            a=np.array([r[metric] for r in rr],float)
            ar=a-X@(np.linalg.pinv(X)@a)
            corr=float(np.corrcoef(ar,yr)[0,1]) if ar.std()>1e-14 else None
            rho=float(spearmanr(a,y).statistic) if a.std()>1e-14 else None
            associations.append(dict(dataset=ds,checkpoint=ck,metric=metric,raw_spearman=rho,adjusted_residual_pearson=corr,n=n))
    return results,associations
def protected():
    files=[B/'picflow_ybranch_smoke_80.npz',P/'pilot.py',P/'freeze.json']
    for m in ['path','branch_tangent']:
        root=(P if m=='branch_tangent' else B/'research_e3_2400')/'runs'/f'{m}_steps2400_seed2026'
        files+=list(root.glob('*.pt'))+list(root.glob('*.json'))
    for root in ['research_e2','research_m4','research_ybranch_branch_tangent_seed2026']:
        files+=list((B/root/'predictions').glob('*.npz'))
        files+=list((B/root).glob('*.py'))+list((B/root).glob('*.json'))+list((B/root).glob('*.csv'))
    files+=list((B/'research_e2').glob('*.npz'))+list((B/'research_m4').glob('*.npz'))
    return {str(p.relative_to(B)):sha(p) for p in sorted(set(files))}
def freeze():
    assert not (O/'freeze.json').exists()
    O.mkdir(exist_ok=True);save('rules.json',RULE)
    save('freeze.json',dict(created_utc=time.time(),rules=RULE,source_sha256=sha(__file__),protocol_sha256=sha(O/'PROTOCOL.md'),
        protected_hashes=protected(),scope='train64 val8 first; E2/M4 retrospective after development lock; old test8 excluded',optimizer_updates=0))
    log('FROZEN')
def verify():
    q=read(O/'freeze.json')
    assert q['source_sha256']==sha(__file__)
    assert q['protocol_sha256']==sha(O/'PROTOCOL.md')
    for n,h in q['protected_hashes'].items():assert sha(B/n)==h,n
    return len(q['protected_hashes'])
def infer():
    import torch
    d=load('DEV');pilot.stage.setup();ff.configure();meta=read(B/'research_e3_2400/manifest.json')
    x,pc=ff.infer_inputs(d['eps'],d['src_mask'],meta['calibration'])
    bt=np.stack([geometry(e,s)['carriers'][1] for e,s in zip(d['eps'],d['src_mask'])])
    cs=dict(path=pc,branch_tangent=torch.from_numpy(bt[:,None]));replays=[]
    (O/'predictions').mkdir(exist_ok=True)
    for method in cs:
      root=(P if method=='branch_tangent' else B/'research_e3_2400')/'runs'/f'{method}_steps2400_seed2026'
      su=read(root/'summary.json')
      for ckname in ['best','last']:
        target=O/'predictions'/f'DEV_{method}_{ckname}.npz'
        assert sha(root/(ckname+'.pt'))==su[ckname+'_sha256']
        if target.exists():
            log('REUSING '+target.name);continue
        ck=torch.load(root/(ckname+'.pt'),weights_only=False,map_location='cpu')
        assert ck['calibration']==meta['calibration'] and ck['scale']==meta['scale']
        model=ff.model_new(2026);model.load_state_dict(ck['state']);model.eval();pred=[]
        with torch.inference_mode():
          for j in range(0,len(x),2):
            v=pilot.stage.e1.ORIGINAL_FORWARD(model,x[j:j+2].cuda(),cs[method][j:j+2].cuda(),'path','fp32')
            pred.append(v[:,0].cpu().numpy()*meta['scale'])
        pred=np.concatenate(pred)
        np.savez_compressed(target,prediction=pred,geometry_id=d['geometry_id'],input_port=d['input_port'])
        # GPU normalized arithmetic used in historical metrics; physical double metrics differ by rounding.
        errs=np.sqrt((abs(pred-d['truth'])**2).sum((1,2))/(abs(d['truth'])**2).sum((1,2)))
        val=errs[d['split']=='val'];train=errs[d['split']=='train']
        expected=np.array(su['best_val_per_sample' if ckname=='best' else 'final_val_per_sample'])
        diff=float(abs(val-expected).max());assert diff<5e-7,(method,ckname,diff)
        exptrain=su['best_train' if ckname=='best' else 'final_train'];assert abs(train.mean()-exptrain)<5e-7
        replays.append(dict(method=method,checkpoint=ckname,val_max_error=diff,train_mean=float(train.mean()),expected_train=exptrain,step=ck['step']))
        log('INFERRED '+method+'/'+ckname+' validation replay '+str(diff))
        del model,ck,v;torch.cuda.empty_cache()
    save('inference_replays.json',replays);verify()
def analyze(ds):
    if ds!='DEV':
        lock=read(O/'development_lock.json');assert lock['source_sha256']==sha(__file__)
    d=load(ds);params=parameters(ds,d);n=len(d['geometry_id'])
    preds={(m,c):prediction(ds,m,c,d['geometry_id']) for m in ['path','branch_tangent'] for c in ['best','last']}
    sample=[];regions=[];phases=[];sensitivity=[];geo=[];maximum=0.;replaymax=0.
    historical={}
    if ds!='DEV':
      with (P/'test_per_geometry.csv').open(encoding='utf-8-sig') as h:
        historical={(r['method'],r['checkpoint'],r['geometry_id']):float(r['rel_l2']) for r in csv.DictReader(h) if r['dataset']==ds}
    # Figure cases fixed by validation/development rule: nearest median delta per port, and max delta adverse, best only.
    losses={}
    for ck in ['best','last']:
        vals=[]
        for i in range(n):
            E=d['truth'][i].astype(np.complex128);den=(abs(E)**2).sum()
            vals.append([np.sqrt((abs(preds[(m,ck)][i].astype(np.complex128)-E)**2).sum()/den) for m in ['path','branch_tangent']])
        losses[ck]=np.array(vals)
    cases={}
    for p in [1,2,3]:
        ix=np.flatnonzero(d['input_port']==p)
        # DEV plots use validation if port exists, otherwise training, explicitly labeled.
        if ds=='DEV' and np.any(d['split'][ix]=='val'):ix=ix[d['split'][ix]=='val']
        if len(ix):
            diff=np.diff(losses['best'][ix],axis=1)[:,0]
            cases[int(ix[np.argmin(abs(diff-np.median(diff)))])]='port'+str(p)+'_median'
    diff=np.diff(losses['best'],axis=1)[:,0];cases[int(np.argmax(diff))]='adverse_max'
    figrecords=[]
    for i in range(n):
        gid=str(d['geometry_id'][i]);group=str(d['split'][i]);port=int(d['input_port'][i])
        ep=d['eps'][i];E=d['truth'][i].astype(np.complex128);I=abs(E)**2;den=I.sum()
        f=geometry(ep,d['src_mask'][i]);masks=partition(f)
        q=burden(E,f);pp=phase(E,f,masks,maps=True);phase_rows,Dmaps,smooth=pp
        q.update(phase_delta_Q=phase_rows[0]['delta_Q'],phase_delta_D=phase_rows[0]['delta_D'],
                 phase_Q_path=phase_rows[0]['Q_path'],phase_Q_bt=phase_rows[0]['Q_bt'],
                 phase_D_path=phase_rows[0]['D_path'],phase_D_bt=phase_rows[0]['D_bt'],
                 smooth_energy_fraction=phase_rows[0]['smooth_energy_fraction'],
                 slope_roughness=float(np.mean(np.diff(f['info']['ty'],n=2,axis=1)**2)))
        ident=dict(dataset=group,geometry_id=gid,input_port=port)
        geo.append(dict(**ident,**params[i],epsmax=float(ep.max()),junction_x_um=f['junction']*.05,**q))
        for pr in phase_rows:phases.append(dict(**ident,threshold=.01,**pr))
        for th in [.001,.05]:
            for pr in phase(E,f,masks,th):phases.append(dict(**ident,threshold=th,**pr))
        for ck in ['best','last']:
            err=[abs(preds[(m,ck)][i].astype(np.complex128)-E)**2/den for m in ['path','branch_tangent']]
            ee=losses[ck][i];row=dict(**ident,checkpoint=ck,e_path=float(ee[0]),e_bt=float(ee[1]),delta_e=float(ee[1]-ee[0]),
                relative_delta=float(ee[1]/ee[0]-1),e2_path=float(ee[0]**2),e2_bt=float(ee[1]**2),delta_e2=float(ee[1]**2-ee[0]**2),
                **params[i],epsmax=float(ep.max()),**q)
            sample.append(row)
            sums=np.zeros(2)
            for r,mask in zip(REG,masks):
                c=np.array([a[mask].sum() for a in err]);sums+=c;energy=I[mask].sum()/den
                regions.append(dict(**ident,checkpoint=ck,region=r,C_path=float(c[0]),C_bt=float(c[1]),delta_C=float(c[1]-c[0]),
                    truth_energy_fraction=float(energy),area_fraction=float(mask.mean()),
                    regional_rel_path=float(np.sqrt(c[0]/energy)) if energy else None,regional_rel_bt=float(np.sqrt(c[1]/energy)) if energy else None))
            maximum=max(maximum,float(abs(sums-ee**2).max()))
            assert np.max(abs(sums-ee**2))<2e-15
            if ds!='DEV':
                for k,m in enumerate(['path','branch_tangent']):
                    replaymax=max(replaymax,abs(ee[k]-historical[(m,ck,gid)]))
            # Sensitivity set fixed before outcomes, one dimension at a time.
            for bw,jx in [(.1,1.),(.4,1.),(.2,.5),(.2,1.5)]:
                mm=partition(f,bw,jx)
                for r,mask in zip(REG,mm):
                    sensitivity.append(dict(**ident,checkpoint=ck,boundary_um=bw,junction_x_half_um=jx,region=r,
                      delta_C=float((err[1]-err[0])[mask].sum())))
            if ck=='best' and i in cases:
                plotcase(ds,group,gid,port,cases[i],E,f,masks,err,Dmaps,smooth)
                figrecords.append(dict(**ident,selection=cases[i],file=f'figures/{ds}_{cases[i]}.png'))
        if i%20==0:log(f'ANALYZED {ds} {i}/{n}')
    csvout(f'{ds}/per_geometry.csv',sample);csvout(f'{ds}/regions.csv',regions);csvout(f'{ds}/phase.csv',phases)
    csvout(f'{ds}/sensitivity.csv',sensitivity);csvout(f'{ds}/geometry_diagnostics.csv',geo)
    save(f'{ds}/figure_cases.json',figrecords)
    save(f'{ds}/verification.json',dict(max_additivity_error=maximum,max_historical_relL2_difference=replaymax,n=n,
        exhaustive_disjoint_masks=True,protected_files_verified=verify(),optimizer_updates=0))
    assert replaymax<5e-7
    aggregate(ds,sample,regions,geo,phases,sensitivity)
    if ds=='DEV':
        assert not (O/'development_lock.json').exists()
        save('development_lock.json',dict(time_utc=time.time(),source_sha256=sha(__file__),rules=RULE,development_summary_sha256=sha(O/'DEV/summary.json'),
          decision_rule='Inspect signed regional excess and energy burden/mismatch together. Prefer single region-limited correction only if both support same region. No intervention executed. E2/M4 retrospective, cannot tune rules.'))
    log('COMPLETE '+ds)
def ci(a):
    a=np.array(a,float);rng=np.random.default_rng(RULE['seed']);b=rng.integers(0,len(a),(RULE['bootstrap'],len(a)))
    return np.quantile(a[b].mean(1),[.025,.975]).tolist()
def aggregate(ds,sample,regions,geo,phases,sensitivity):
    summary=[];regsum=[];geosum=[];phsum=[];sensum=[]
    for group in sorted(set(r['dataset'] for r in sample)):
      for port in [0,1,2,3]:
        gg=[r for r in geo if r['dataset']==group and (not port or r['input_port']==port)]
        if not gg:continue
        ident=dict(dataset=group,input_port=port,n=len(gg))
        for key in ['union_J_path','union_J_bt','union_B_path','union_B_bt','union_delta_B','union_A_jump_path','union_A_jump_bt',
                    'union_edge_energy_fraction','branch_delta_B','route_delta_B','phase_Q_path','phase_Q_bt','phase_delta_Q',
                    'phase_D_path','phase_D_bt','phase_delta_D','smooth_energy_fraction','weighted_correction_phase_rms','slope_roughness']:
            vals=[r[key] for r in gg if r.get(key) is not None]
            geosum.append(dict(**ident,metric=key,mean=float(np.mean(vals)),median=float(np.median(vals)),ci95=ci(vals)))
        for ck in ['best','last']:
          rr=[r for r in sample if r['dataset']==group and r['checkpoint']==ck and (not port or r['input_port']==port)]
          summary.append(dict(**ident,checkpoint=ck,**{k:float(np.mean([r[k] for r in rr])) for k in ['e_path','e_bt','delta_e','relative_delta','e2_path','e2_bt','delta_e2']},
                relative_of_means=float(np.mean([r['e_bt'] for r in rr])/np.mean([r['e_path'] for r in rr])-1),
                delta_e_ci95=ci([r['delta_e'] for r in rr]),worse=int(sum(r['delta_e']>0 for r in rr))))
          total=np.mean([r['delta_e2'] for r in rr])
          for reg in REG:
            rrr=[r for r in regions if r['dataset']==group and r['checkpoint']==ck and r['region']==reg and (not port or r['input_port']==port)]
            delta=np.mean([r['delta_C'] for r in rrr])
            regsum.append(dict(**ident,checkpoint=ck,region=reg,**{k:float(np.mean([r[k] for r in rrr])) for k in ['C_path','C_bt','delta_C','truth_energy_fraction','area_fraction']},
                signed_fraction_total_excess=float(delta/total) if total else None,ci95=ci([r['delta_C'] for r in rrr])))
          for bw,jx in [(.1,1.),(.4,1.),(.2,.5),(.2,1.5)]:
            for reg in REG:
                ss=[r['delta_C'] for r in sensitivity if r['dataset']==group and r['checkpoint']==ck and r['region']==reg and r['boundary_um']==bw and r['junction_x_half_um']==jx and (not port or r['input_port']==port)]
                sensum.append(dict(**ident,checkpoint=ck,boundary_um=bw,junction_x_half_um=jx,region=reg,mean_delta_C=float(np.mean(ss))))
        for th in [.001,.01,.05]:
          for reg in ['all']+REG:
            rr=[r for r in phases if r['dataset']==group and r['threshold']==th and r['region']==reg and (not port or r['input_port']==port)]
            phsum.append(dict(**ident,threshold=th,region=reg,**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None
              for k in ['D_path','D_bt','delta_D','Q_path','Q_bt','delta_Q','smooth_energy_fraction']}))
    adj,assoc=fit_controls(sample)
    save(f'{ds}/summary.json',dict(overall=summary,regions=regsum,geometry=geosum,phase=phsum,adjusted_ports=adj,associations=assoc,sensitivity=sensum))
    for name,rows in [('overall',summary),('regional_summary',regsum),('carrier_summary',geosum),('phase_summary',phsum),
                      ('adjusted_ports',adj),('associations',assoc),('sensitivity_summary',sensum)]:csvout(f'{ds}/{name}.csv',rows)
def plotcase(ds,group,gid,port,label,E,f,masks,err,Dmaps,smooth):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm,ListedColormap
    out=O/'figures';out.mkdir(exist_ok=True)
    extent=[0,E.shape[1]*.05,0,E.shape[0]*.05]
    fig,ax=plt.subplots(4,1,figsize=(12,9.5),sharex=True,sharey=True)
    im=ax[0].imshow(abs(E),origin='lower',extent=extent,aspect='auto',cmap='magma')
    ax[0].contour(f['core'],[.5],colors='cyan',linewidths=.35,origin='lower',extent=extent)
    ax[0].contour(f['seam'],[.5],colors='lime',linewidths=.6,origin='lower',extent=extent)
    ax[0].set_title('Truth |E|; cyan=material; green=branch / routing seams',fontsize=10);fig.colorbar(im,ax=ax[0],pad=.01)
    regmap=np.zeros(E.shape,int)
    for j,mask in enumerate(masks):regmap[mask]=j
    cmap=ListedColormap(['#f4a261','#9b5de5','#e63946','#457b9d','#8ecae6','#eeeeee'])
    im=ax[1].imshow(regmap,origin='lower',extent=extent,aspect='auto',cmap=cmap,vmin=-.5,vmax=5.5)
    cb=fig.colorbar(im,ax=ax[1],pad=.01,ticks=np.arange(6));cb.ax.set_yticklabels(REG,fontsize=7)
    ax[1].set_title('Disjoint regions: junction > seam band > cores > cladding',fontsize=10)
    diff=err[1]-err[0];lim=max(float(np.quantile(abs(diff),.995)),1e-10)
    im=ax[2].imshow(diff,origin='lower',extent=extent,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-lim,lim))
    ax[2].contour(f['core'],[.5],colors='k',linewidths=.3,origin='lower',extent=extent)
    ax[2].contour(f['seam'],[.5],colors='lime',linewidths=.6,origin='lower',extent=extent)
    ax[2].set_title('BT minus Path squared-error contribution per pixel; red=worse; clipped at p99.5',fontsize=10);fig.colorbar(im,ax=ax[2],pad=.01)
    dv=np.ma.masked_where(~smooth,Dmaps[1]-Dmaps[0]);lim=max(float(np.quantile(abs(dv.compressed()),.99)),1e-8)
    im=ax[3].imshow(dv,origin='lower',extent=extent,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-lim,lim))
    ax[3].contour(f['core'],[.5],colors='k',linewidths=.3,origin='lower',extent=extent)
    ax[3].set_title('Smooth-region local phase-gradient mismatch change; gray/white=excluded; rad^2/pixel^2',fontsize=10)
    fig.colorbar(im,ax=ax[3],pad=.01)
    for a in ax:a.set_ylabel('y (um)')
    ax[-1].set_xlabel('x (um)')
    fig.suptitle(f'{ds} ({group}), port {port}, {gid}, {label}; one frozen seed 2026',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(out/f'{ds}_{label}.png',dpi=170);plt.close(fig)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','infer','DEV','E2','M4','verify']);args=a.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='infer':infer()
    elif args.action=='verify':log(verify())
    else:analyze(args.action)
