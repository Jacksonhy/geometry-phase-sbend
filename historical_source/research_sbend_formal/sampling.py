"""Geometry-only PCG64 stream, width stratification, equivalent exclusions."""
import numpy as np,hashlib,json
from common import OUT,V1,DEV,v1,read,write,params_key,curve_fingerprint,SEED
def development_exclusions():
    rows=[]
    for r in read(DEV/'sbend_index.json'):
        poly,_=v1.centerline_from_params(r['params'])
        rows.append({'geometry_id':r['geometry_id'],'parameter_key':params_key(r['params']),
            'centerline_equivalence_sha256':curve_fingerprint(poly)})
    return rows
def feasible(params):
    try:
        poly,bend=v1.centerline_from_params(params)
        margin=params['wg_width_um']/2+.5
        ok=abs(bend[:,0]).max()+margin<12 and abs(bend[:,1]).max()+margin<4
        return bool(ok),poly,bend
    except ValueError:return False,None,None
def generate():
    assert read(OUT/'preflight.json')['passed']
    assert not (OUT/'geometry_manifest.json').exists(),'Do not resample formal manifest'
    rng=np.random.default_rng(SEED);dev=development_exclusions();used={tuple(r['parameter_key']) for r in dev}
    curves={(r['parameter_key'][0],r['centerline_equivalence_sha256']) for r in dev};records=[];rejected=[];draw=0
    for wi in range(16,24):
        for split,n in [('train',8),('val',2),('test',13 if wi<20 else 12)]:
            accepted=[]
            while len(accepted)<n:
                draw+=1;oi=int(rng.integers(0,141));ri=int(rng.integers(0,161));key=(wi,80+oi,120+ri)
                params=dict(zip(['wg_width_um','lateral_offset_um','R_min_um'],[a*.025 for a in key]))
                ok,poly,bend=feasible(params);reason=None
                # Same centerline with different width is allowed: shape includes width.
                cf=curve_fingerprint(poly) if ok else None;shape=(wi,cf)
                if key in used:reason='parameter_duplicate_or_development'
                elif not ok:reason='geometric_fit'
                elif shape in curves:reason='centerline_width_equivalent'
                if reason:rejected.append({'draw':draw,'parameter_key':key,'reason':reason});continue
                used.add(key);curves.add(shape)
                canon=json.dumps(dict(zip(['w_grid','offset_grid','R_grid'],key)),sort_keys=True,separators=(',',':'))
                gid='sbend_formal_'+hashlib.sha256(canon.encode()).hexdigest()[:16]
                accepted.append({'geometry_id':gid,'params':params,'parameter_key':key,'split':split,'draw':draw,
                    'centerline_equivalence_sha256':cf})
            dirs=[1]*(n//2)+[2]*(n//2)
            if n%2:dirs.append(1 if (wi-16)%2==0 else 2)
            rng.shuffle(dirs)
            for r,ip in zip(accepted,dirs):r['input_port']=int(ip)
            order=rng.permutation(n)
            records.extend(accepted[j] for j in order)
    assert len(records)==180 and len({r['geometry_id'] for r in records})==180
    devshapes={(r['parameter_key'][0],r['centerline_equivalence_sha256']) for r in dev}
    assert not any((r['parameter_key'][0],r['centerline_equivalence_sha256']) in devshapes for r in records)
    for split,n,left in [('train',64,32),('val',16,8),('test',100,50)]:
        rr=[r for r in records if r['split']==split];assert len(rr)==n and sum(r['input_port']==1 for r in rr)==left
    write('geometry_manifest.json',{'seed':SEED,'rng':'NumPy PCG64','records':records,'development_exclusions':dev,
        'rejections':rejected,'eps_duplicate_check':'post-FDTD float32 exact / axis flips / 180 rotation before training',
        'test_fields_read_for_selection':False})
if __name__=='__main__':generate()
