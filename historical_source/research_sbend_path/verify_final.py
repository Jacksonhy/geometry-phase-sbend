"""Independently check frozen sources, saved phase calculations and history."""
import ast,hashlib,inspect,re
import numpy as np
from common import ROOT,OUT,OLD,read,save,sha,now
from path_geometry import centerline_from_params,distances_at_points,geometry_features,smooth_tangent_at_x
def main():
    changed=[];before=read(OUT/'historical_before.json')
    shards=read(OUT/'shards_before.json');sbad=[]
    for name,entry in shards.items():
        p=ROOT/'picflow_source/shards'/name
        if not p.exists() or p.stat().st_size!=entry['bytes'] or p.stat().st_mtime_ns!=entry['mtime_ns']:sbad.append(name)
    freeze=read(OUT/'frozen_protocol.json');fbad=[rel for rel,h in freeze['sources'].items() if sha(ROOT/rel)!=h]
    assert not fbad,fbad
    checks=[]
    for rec in read(OLD/'sbend_index.json'):
        gid=rec['geometry_id'];p=rec['params'];poly,bend=centerline_from_params(p)
        with np.load(OLD/rec['sample'],allow_pickle=False) as d,np.load(OUT/'maps'/f'{gid}.npz') as m,np.load(OUT/'profiles'/f'{gid}.npz') as prof:
            src=d['source_center_um'];sy=float(d['source_size_um'][1]);ip=int(d['input_port'])
            points=np.column_stack([np.full(101,src[0]),np.linspace(src[1]-sy/2,src[1]+sy/2,101)])
            f=distances_at_points(points,poly,src,ip)
            err=max(float(abs(f[name]).max()) for name in ('linear','column_arc','path'));assert err<1e-10
            # LocalTangent is an approximate normal extension. Interpolating the
            # vertex tangent leaves a tiny source-plane tilt on the left lead;
            # do not incorrectly demand exact plane constancy from this baseline.
            lt_plane=float(abs(f['local_tangent']).max())
            ty=float(smooth_tangent_at_x(np.asarray([src[0]]),poly)[0,1])
            lt_expected=f['direction']*ty*(points[:,1]-src[1])
            assert np.max(abs(f['local_tangent']-lt_expected))<1e-10
            at_source=distances_at_points(src[None,:2],poly,src,ip)
            assert max(abs(float(at_source[k][0])) for k in ('linear','column_arc','local_tangent','path'))<1e-10
            # Phase metrics from cached samples: direct complex envelope ratios = analytic phase subtraction.
            E=prof['Ez_samples'];phi=prof['path_phase'];envelope=E*np.exp(-1j*phi)
            g=np.angle(envelope[1:]*envelope[:-1].conj())/.05
            direct=np.angle(E[1:]*E[:-1].conj()*np.exp(-1j*np.diff(phi,axis=0)))/.05
            error=float(np.nanmax(abs(g-direct)));assert error<1e-10
            assert np.max(abs(m['path_carrier']-np.exp(1j*m['beta_Y_rad_um']*m['path_um'])))<1e-7
            # Translation invariance of source anchoring on all straight points, no mask centroid involved.
            shift=np.array([.7,-.2]);q=poly[::29];s1=distances_at_points(q,poly,src,ip)['path'];ss=src.copy();ss[:2]+=shift
            s2=distances_at_points(q+shift,poly+shift,ss,ip)['path'];assert np.max(abs(s1-s2))<1e-10
            checks.append({'geometry_id':gid,'path_linear_column_arc_source_plane_max_um':err,
                'local_tangent_source_plane_tilt_max_um':lt_plane,'local_tangent_source_plane_tilt_phase_max_rad':lt_plane*float(m['beta_Y_rad_um']),
                'all_carriers_source_center_zero':True,'saved_residual_reconstruction_error_rad_um':error,
                'translation_equivariance':True,'carrier_matches_geometry':True})
    # Code contract: path adapter reads only official source function; no target arrays accepted.
    params=list(inspect.signature(geometry_features).parameters)
    assert params==['params','x_um','y_um','source_center_um','input_port']
    report=OUT/'S-bend路径适配与物理诊断报告.md';missing=[]
    for target in re.findall(r'\]\(<([^>]+)>\)',report.read_text()):
        if target.startswith('D:/game/'):
            rel=re.sub(r':\d+$','',target[len('D:/game/'):])
            if rel in ['research_sbend_path/verification.json','research_sbend_path/outputs_manifest.json']:continue
            if not (ROOT/rel).exists():missing.append(rel)
    assert not sbad and not missing,(sbad,missing)
    assert read(OUT/'geometry_verification.json')['passed']
    assert len(read(OUT/'development_exclusions.json')['records'])==16
    assert freeze['formal_samples_instantiated']==0
    for rel,entry in before.items():
        p=ROOT/rel
        if not p.is_file() or sha(p)!=entry['sha256']:changed.append(rel)
    assert not changed,changed
    result={'time_utc':now(),'passed':True,'historical_files_hashed':len(before),'historical_changes':changed,'shard_stat_checked':len(shards),
        'shard_changes':sbad,'frozen_source_changes':fbad,'development_geometries_checked':16,'checks':checks,
        'formal_geometries_generated':0,'fdtd_runs_this_turn':0,'neural_training_or_inference':False,'new_formal_test_evaluations':0,
        'missing_report_links':missing,'notes':'All physics/phase checks use previously seen development fields; no blindness claim.'}
    save('verification.json',result)
    save('outputs_manifest.json',{str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and p.name!='outputs_manifest.json'})
    print(str({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':main()
