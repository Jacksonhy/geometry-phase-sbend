"""Independent saved-prediction replay. No model inference or optimization."""
import csv,sys
import numpy as np
from common import *

def main():
    verify_freeze();complete=read(OUT/'evaluation_complete.json');training=read(OUT/'training_complete.json')
    assert complete['one_test_event'] and complete['geometries']==100
    before=read(OUT/'historical_before.json');changed=[]
    for name,e in before.items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=e['sha256']:changed.append(name)
    shards=read(OUT/'shards_before.json');sc=[]
    for name,e in shards.items():
        p=ROOT/'picflow_source/shards'/name
        if not p.exists() or p.stat().st_size!=e['bytes'] or p.stat().st_mtime_ns!=e['mtime_ns']:sc.append(name)
    assert not changed and not sc,(changed,sc)
    for name,h in training['weight_hashes'].items():assert sha(OUT/name)==h
    for name,h in complete['prediction_hashes'].items():assert sha(OUT/name)==h
    for name,h in complete['metric_files_sha256'].items():assert sha(OUT/name)==h
    assert sha(OUT/'geometry_manifest.json')==read(OUT/'geometry_freeze.json')['manifest_sha256']
    data=read(OUT/'data_complete.json')
    for r in data['records']:assert sha(OUT/'data'/r['geometry_id']/'sample.npz')==r['sample_sha256']
    recs=[r for r in read(OUT/'geometry_manifest.json')['records'] if r['split']=='test']
    with (OUT/'test_per_geometry.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    lookup={(r['method'],r['checkpoint'],r['geometry_id']):r for r in rows};maxdiff=0.;checks=0
    for name in complete['prediction_hashes']:
        path=OUT/name;stem=path.stem;method,checkpoint=stem.rsplit('_',1)
        with np.load(path,allow_pickle=False) as p:
            assert p['geometry_id'].tolist()==[r['geometry_id'] for r in recs]
            for i,r in enumerate(recs):
                with np.load(OUT/'data'/r['geometry_id']/'sample.npz',allow_pickle=False) as z:truth=z['Ez'].astype(np.complex128)
                delta=p['prediction'][i].astype(np.complex128)-truth
                err=abs(delta)**2;power=abs(truth)**2;values={'rel_l2':float(np.sqrt(np.sum(err)/np.sum(power)))}
                with np.load(OUT/'cache'/(r['geometry_id']+'.npz'),allow_pickle=False) as z:
                    for key in ['core','bend_core','halo','far']:
                        mask=z['region_'+key];values[key+'_rel_l2']=float(np.sqrt(err[mask].sum()/power[mask].sum()))
                for k,v in values.items():
                    diff=abs(v-float(lookup[method,checkpoint,r['geometry_id']][k]));maxdiff=max(maxdiff,diff);assert diff<1e-10;checks+=1
    assert len(lookup)==1000
    write('verification.json',{'time_utc':now(),'passed':True,'history_files_hashed':len(before),'historical_changes':changed,
        'shards_stat_checked':len(shards),'shard_changes':sc,'formal_samples_hashed':180,'weight_files_hashed':len(training['weight_hashes']),
        'predictions_hashed':10,'metric_checks':checks,'metric_max_difference':maxdiff,
        'new_inference_or_training':False,'test_event_count':1,'single_training_seed_only':2026})
    print('SAVED RESULTS AND HISTORICAL INTEGRITY VERIFIED',flush=True)
if __name__=='__main__':main()
