"""Only synthetic / existing 16 development data; no optimizer updates."""
import sys,inspect,time,traceback
import numpy as np
import torch
from common import *
from features import geometry,inputs,pack,unpack,sample_load
from sampling import development_exclusions,feasible
from calibration import extract
import model

def main():
    assert not (OUT/'preflight.json').exists()
    for f,base in [(V1/'frozen_protocol.json',ROOT),(V2/'protocol_freeze.json',V2),(ROOT/'research_sbend_path_v3/protocol_freeze.json',ROOT/'research_sbend_path_v3')]:
        for n,h in read(f)['sources'].items():assert sha(base/n)==h,'Historic freeze changed: '+n
    model.configure();rng=np.random.default_rng(SEED);a=rng.normal(size=(2,12,160,480)).astype(np.float32)
    assert np.array_equal(a,unpack(pack(a)))
    ex=development_exclusions();assert len(ex)==16 and len({tuple(r['parameter_key']) for r in ex})==16
    checkrows=[];zlist=[];clist=[];ylist=[];slopes=[]
    for i,r in enumerate(read(DEV/'sbend_index.json')):
        d=sample_load(DEV/r['sample']);p=r['params'];ip=int(d['input_port']);x=d['x_um'];y=d['y_um'];src=d['source_center_um']
        assert d['Ez'].shape==(160,480) and abs(src[0]-(-11.9 if ip==1 else 11.9))<1e-12
        ok,poly,bend=feasible(p);assert ok
        ds,cost,info=geometry(p,x,y,src,ip);assert cost['v2_max_axis_gradient']<=5
        points=np.column_stack([np.full(17,src[0]),np.linspace(-4,4,17)])
        ref1=v1.distances_at_points(points,poly,src,ip)
        ref2=v2.distances_at_points(points,poly,src,ip,p['wg_width_um'])
        sourceerrors={n:float(abs(ref1[n]).max()) for n in ['linear','path','local_tangent']}
        sourceerrors['path_v2']=float(abs(ref2['path_v2']).max())
        assert max(sourceerrors[n] for n in ['linear','path','path_v2'])<1e-10
        assert sourceerrors['local_tangent']<1e-4 # known frozen interpolation effect; do not alter baseline
        beta=float(np.polyval(read(ROOT/'research_e3_2400/manifest.json')['calibration'],d['eps'].max())/.05)
        z,c=inputs(d['eps'],d['src_mask'],x,y,src,ip,beta,ds)
        errors={m:float(abs((d['Ez']/c[m][0])*c[m][0]-d['Ez']).max()) for m in METHODS}
        assert max(errors.values())<1e-5
        modeerrors=max(float(abs(abs(q)-1).max()) for q in c.values());assert modeerrors<2e-7
        epsback=z[0].real*4.2+2.0851359;assert np.max(abs(epsback-d['eps']))<1e-6
        q=extract(d['Ez'],x,y,p,ip);slopes.append({'geometry_id':r['geometry_id'],**q})
        checkrows.append({'geometry_id':r['geometry_id'],'input_port':ip,**cost,'source_errors_um':sourceerrors,
            'carrier_unit_modulus_error':modeerrors,'field_roundtrip_max':max(errors.values())})
        if i in [0,1]:zlist.append(z);clist.append(c);ylist.append(d['Ez'][None].astype(np.complex64))
    x=torch.from_numpy(np.stack(zlist)).cuda();y=torch.from_numpy(np.stack(ylist)).cuda()
    records=[];initials=[];modeloutputs=[]
    for name in METHODS:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=model.new();initial=model.state_hash(m);initials.append(initial)
        m.eval();c=torch.from_numpy(np.stack([a[name] for a in clist])).cuda()
        eps=x[:,0:1]*4.2+2.0851359
        prior=m.get_grid((2,14,160,480),x.device,mode='exp',epsilon=eps,
            wavelength=torch.full((2,1),1.55,device='cuda'),grid_step=torch.full((2,2),.05,device='cuda'))
        assert torch.isfinite(prior).all()
        before=time.perf_counter();pred=model.forward(m,x,c);loss=model.rel_mse(pred,y).mean();assert torch.isfinite(loss)
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        torch.cuda.synchronize();elapsed=time.perf_counter()-before
        assert model.state_hash(m)==initial,'No parameter/buffer updates allowed in preflight'
        envelope=(pred/c).detach().cpu().numpy();modeloutputs.append(envelope)
        records.append({'method':name,'initial_state_sha256':initial,'batch_schedule_sha256':ahash(model.batches()),
            'parameters':sum(p.numel() for p in m.parameters()),'real_scalar_degrees':sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters()),
            'output_shape':list(pred.shape),'prior_sha256':ahash(prior.detach().cpu().numpy()),
            'forward_backward_seconds':elapsed,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2,
            'optimizer_updates':0})
        del m,pred,loss,c,prior
    assert len(set(initials))==1 and len({a['prior_sha256'] for a in records})==1
    assert max(float(abs(a-modeloutputs[0]).max()) for a in modeloutputs)<2e-6
    assert len(set(a['parameters'] for a in records))==1
    result={'time_utc':now(),'passed':True,'development_geometries':16,'synthetic_pack_roundtrip':True,
       'geometry_checks':checkrows,'model_checks':records,'calibration_extractor_development_checks':slopes,
       'formal_beta_fitted':False,'optimizer_updates':0,'new_geometry_or_field_used':False,
       'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
       'numpy':np.__version__,'python':sys.version,'model_config':model.CONFIG,'development_exclusions':ex}
    write('preflight.json',result);print('PREFLIGHT PASSED',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        write('preflight_failure.json',{'time_utc':now(),'exception':repr(exc),'traceback':traceback.format_exc(),'formal_generation_started':False})
        raise
