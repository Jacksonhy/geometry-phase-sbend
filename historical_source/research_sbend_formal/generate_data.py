"""Serial FDTD and strictly structural QC; no test performance inspection."""
import os,sys,time,subprocess,traceback
from common import *

def qc(rec):
    p=OUT/'data'/rec['geometry_id'];meta=read(p/'result.json')
    with np.load(p/'sample.npz',allow_pickle=False) as z:
        E=z['Ez'];ep=z['eps'];x=z['x_um'];y=z['y_um'];ip=rec['input_port'];src=z['source_center_um']
        assert str(z['geometry_id'])==rec['geometry_id'] and str(z['split'])==rec['split']
        assert E.shape==ep.shape==(160,480)
        assert all(np.isfinite(z[k]).all() for k in ['Ez','Hx','Hy','eps','alpha_native','S_native'])
        assert np.max(abs(E))>1e-6 and meta['qc']['passed']
        assert np.allclose(np.diff(x),.05,rtol=0,atol=1e-12) and np.allclose(np.diff(y),.05,rtol=0,atol=1e-12)
        assert float(z['wavelength_um'])==1.55 and np.array_equal(z['port_ids'],[1,2])
        assert all(float(z['params/'+k])==v for k,v in rec['params'].items())
        assert abs(src[0]-(-11.9 if ip==1 else 11.9))<1e-12
        poly,bend=v1.centerline_from_params(rec['params']);assert abs(src[1]-bend[0 if ip==1 else -1,1])<1e-10
        X,Y=np.meshgrid(x,y);sy=z['source_size_um'][1]
        expected=(X-src[0])**2+(Y-np.clip(Y,src[1]-sy/2,src[1]+sy/2))**2<=.15**2+1e-14
        assert np.array_equal(z['src_mask']>0,expected)
        line=z['src_plane_interpolation_weights'];assert abs(float(np.sum(line*x[None,:])/line.sum())-src[0])<1e-12
        a=z['alpha_native'];S=a[np.arange(2),[1,0]]/a[ip-1,0 if ip==1 else 1]
        assert np.allclose(S,z['S_native'],rtol=0,atol=1e-14)
        assert np.sum(abs(S)**2)<=1.5
        fps=sorted(eps_fingerprints(ep))
    result={'geometry_id':rec['geometry_id'],'split':rec['split'],'passed':True,
        'sample_sha256':sha(p/'sample.npz'),'eps_equivalence_sha256':fps,
        'test_QC_only':rec['split']=='test','phase_performance_inspected':False}
    write(p/'verification.json',result);return result

def run():
    verify_freeze();manifest=read(OUT/'geometry_manifest.json')
    if not (OUT/'geometry_freeze.json').exists():
        write('geometry_freeze.json',{'time_utc':now(),'manifest_sha256':sha(OUT/'geometry_manifest.json'),
             'code_freeze_sha256':sha(OUT/'code_freeze.json'),'before_first_fdtd':True})
    assert sha(OUT/'geometry_manifest.json')==read(OUT/'geometry_freeze.json')['manifest_sha256']
    seen=set()
    for r in read(DEV/'sbend_index.json'):
        with np.load(DEV/r['sample'],allow_pickle=False) as z:seen.update(eps_fingerprints(z['eps']))
    results=[]
    (OUT/'logs').mkdir(exist_ok=True)
    py='/home/cudauser/.conda/envs/picflow-meep/bin/python'
    for i,r in enumerate(manifest['records']):
        dest=OUT/'data'/r['geometry_id']
        if (dest/'verification.json').exists():
            rec=read(dest/'verification.json');assert sha(dest/'sample.npz')==rec['sample_sha256']
        else:
            assert not dest.exists(),'Incomplete FDTD sample; manual diagnosis required, no overwrite'
            log=OUT/'logs'/f'fdtd_{i:03d}.log';t=time.monotonic();peak=0;reason=None
            with log.open('x') as f:
                child=subprocess.Popen([py,'-B','-u',str(OUT/'fdtd_worker.py'),'sbend',str(i)],stdout=f,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
                while child.poll() is None:
                    try:
                        status=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                        rss=int(next(s.split()[1] for s in status if s.startswith('VmRSS:')))/1024;peak=max(peak,rss)
                    except (FileNotFoundError,StopIteration,ProcessLookupError):pass
                    if time.monotonic()-t>900:reason='wall_seconds>900'
                    if peak>4096:reason='RSS>4GiB'
                    if reason:
                        child.terminate()
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(.5)
            resource={'index':i,'geometry_id':r['geometry_id'],'split':r['split'],'wall_seconds':time.monotonic()-t,
                'peak_rss_MiB':peak,'exit_code':child.returncode,'stop_reason':reason}
            write(OUT/'logs'/f'fdtd_{i:03d}_resource.json',resource)
            if child.returncode or reason:raise RuntimeError('FDTD failed: '+str(resource))
            rec=qc(r)
        assert not (set(rec['eps_equivalence_sha256'])&seen),'Exact/axis-reflected eps duplicate; pause'
        seen.update(rec['eps_equivalence_sha256']);results.append(rec)
        write('generation_progress.json',{'completed':len(results),'total':180,'last_geometry_id':r['geometry_id'],'time_utc':now()})
        print(f'FDTD/QC {len(results)}/180 {r["split"]} {r["geometry_id"]}',flush=True)
    write('data_complete.json',{'time_utc':now(),'count':180,'all_structural_QC_passed':True,'no_equivalent_eps_duplicates':True,
       'records':results,'test_phase_or_model_performance_inspected':False})
if __name__=='__main__':
    try:run()
    except Exception as e:
        write('generation_failure.json',{'time_utc':now(),'error':repr(e),'traceback':traceback.format_exc(),'paused':True});raise
