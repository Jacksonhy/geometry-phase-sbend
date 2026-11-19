import inspect,re
import numpy as np
from support import ROOT,OUT,V1,OLD,read,save,sha,now
import path_v2 as v2

def main():
    freeze=read(OUT/'candidate_freeze.json')
    candidate_changed=[n for n,h in freeze['sources'].items() if sha(OUT/n)!=h]
    assert not candidate_changed,candidate_changed
    oldfreeze=read(V1/'frozen_protocol.json')
    old_changed=[n for n,h in oldfreeze['sources'].items() if sha(ROOT/n)!=h]
    assert not old_changed,old_changed
    assert freeze['candidate_count']==1 and read(OUT/'decision.json')['passes_declared_checks']
    assert read(OUT/'refinement_summary.json')['nonfinite_count']==0
    assert all(r['passed'] for r in read(OUT/'geometry_invariants.json'))
    assert list(inspect.signature(v2.geometry_features).parameters)==['params','x_um','y_um','source_center_um','input_port']
    records=[]
    for rec in read(OLD/'sbend_index.json'):
        gid=rec['geometry_id'];ip=rec['input_port'];w=rec['params']['wg_width_um']
        with (np.load(OUT/'maps'/f'{gid}.npz') as m,np.load(V1/'maps'/f'{gid}.npz') as old,
            np.load(OUT/'profiles'/f'{gid}.npz') as p,np.load(V1/'profiles'/f'{gid}.npz') as oldp):
            q=np.stack(np.meshgrid(m['x_um'][::19],m['y_um'][::13]),axis=-1)
            d=v2.distances_at_points(q,old['polyline_um'],old['source_center_um'],ip,w)['path_v2']
            maperr=float(abs(d-m['path_v2_um'][::13,::19]).max())
            beta=float(m['beta_Y_rad_um']);carriererr=float(abs(m['path_v2_carrier']-np.exp(1j*beta*m['path_v2_um'])).max())
            E=oldp['Ez_samples'];phi=p['path_v2_phase'];U=E*np.exp(-1j*phi)
            direct=np.angle(E[1:]*E[:-1].conj()*np.exp(-1j*np.diff(phi,axis=0)))/.05
            residual=np.angle(U[1:]*U[:-1].conj())/.05
            err=float(np.max(abs(direct-residual)))
            assert maperr<1e-9 and carriererr<1e-7 and err<1e-9,(gid,maperr,carriererr,err)
            records.append({'geometry_id':gid,'map_spot_error_um':maperr,'saved_carrier_error':carriererr,'residual_reconstruction_error':err})
    report=OUT/'Path_v2连续性与物理诊断报告.md';missing=[]
    for target in re.findall(r'\]\(<([^>]+)>\)',report.read_text(encoding='utf-8')):
        if target.startswith('D:/game/'):
            rel=target[len('D:/game/'):]
            if rel in ['research_sbend_path_v2/verification.json','research_sbend_path_v2/outputs_manifest.json']:continue
            if not (ROOT/rel).exists():missing.append(rel)
    assert not missing,missing
    before=read(OUT/'historical_before.json');changed=[]
    for rel,entry in before.items():
        p=ROOT/rel
        if not p.is_file() or sha(p)!=entry['sha256']:changed.append(rel)
    shards=read(OUT/'shards_before.json');shardchanged=[]
    for n,e in shards.items():
        p=ROOT/'picflow_source/shards'/n
        if not p.exists() or p.stat().st_size!=e['bytes'] or p.stat().st_mtime_ns!=e['mtime_ns']:shardchanged.append(n)
    assert not changed and not shardchanged,(changed,shardchanged)
    save('protocol_freeze.json',{'time_utc':now(),'candidate_count':1,'methods':['Direct','Linear','LocalTangent','Path_v1','Path_v2'],
        'sources':{n:sha(OUT/n) for n in ['PROTOCOL.md','path_v2.py','analyze.py','refine_geometry.py','FORMAL_PROTOCOL_ADDENDUM.md','candidate_freeze.json']},
        'formal_geometry_counts':{'train':64,'val':16,'test':100},'formal_geometries_generated':0,
        'training_executor_frozen':False,'remaining_preflight':'12 common real channel packing, shared prior and no-training compatibility check',
        'development_exclusion_source_sha256':sha(V1/'development_exclusions.json'),'neural_runs':0,'fdtd_runs':0,'new_test_evaluations':0})
    result={'time_utc':now(),'passed':True,'candidate_count':1,'historical_files_hashed':len(before),'historical_changes':changed,
        'shards_stat_checked':len(shards),'shard_changes':shardchanged,'v1_frozen_source_changes':old_changed,
        'candidate_source_changes_since_pre_diagnostic_freeze':candidate_changed,'development_geometries_checked':len(records),
        'checks':records,'missing_report_links':missing,'neural_runs':0,'fdtd_runs':0,'formal_geometries_generated':0,'new_test_evaluations':0}
    save('verification.json',result)
    save('outputs_manifest.json',{str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and p.name!='outputs_manifest.json'})
    print({k:v for k,v in result.items() if k!='checks'},flush=True)
if __name__=='__main__':main()
