"""Independent replay of predefined primary bootstrap from saved CSV."""
import csv
from common import *
def main():
    with (OUT/'test_per_geometry.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    lookup={(r['method'],r['checkpoint'],r['geometry_id']):float(r['rel_l2']) for r in rows}
    checks=[];pairs=read(OUT/'paired_comparisons.json')
    for p in pairs:
        if p['group']!='all' or p['metric']!='rel_l2':continue
        cp=p['checkpoint'];gids=sorted(g for m,c,g in lookup if m=='path_v2' and c==cp)
        a=np.array([lookup[p['baseline'],cp,g] for g in gids]);b=np.array([lookup['path_v2',cp,g] for g in gids])
        d=a-b;rng=np.random.Generator(np.random.PCG64(2026));indices=rng.integers(0,100,size=(10000,100))
        ci=np.percentile(np.mean(d[indices],axis=1),[2.5,97.5])
        assert np.max(abs(ci-p['bootstrap_geometry_ci95']))<1e-12
        assert abs(d.mean()-p['mean_baseline_minus_v2'])<1e-12
        assert int(np.count_nonzero(d>0))==p['improved_geometries']
        checks.append({'checkpoint':cp,'baseline':p['baseline'],'mean':float(d.mean()),'ci95':ci.tolist(),'passed':True})
    write('statistics_verification.json',{'time_utc':now(),'passed':True,'primary_comparisons_replayed':len(checks),
        'checks':checks,'new_inference':False,'bootstrap_unit':'geometry','training_seed_variability_verified':False})
    print('PRIMARY GEOMETRY BOOTSTRAP REPLAY PASSED',flush=True)
if __name__=='__main__':main()
