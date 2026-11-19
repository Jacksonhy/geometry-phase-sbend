"""Independent scalar circular residual check at fixed physical points."""
import cmath,math
import numpy as np
from experiment import R,read,save,prior
from analyze import RECORDS,A,load,sbend,interpolate,incident
checks=[];bet={r['job_id']:r for r in read(R/'beta.json')}
for r in [RECORDS[0],RECORDS[4],RECORDS[8]]:
 gid=r['geometry_id'];a0=load(prior(gid,20));poly,bend,masks=A.geometry(r,a0);X,Y=np.meshgrid(a0['x_um'],a0['y_um']);pts=np.stack([X,Y],axis=-1);al=sbend(gid,30);ah=sbend(gid,40);el=interpolate(al,pts)/incident(al);eh=interpolate(ah,pts)/incident(ah);mask=masks['bend_core']&(abs(eh)>=.05*abs(eh[masks['bend_core']]).max());pv=pts[mask];lo=el[mask];hi=eh[mask];ip=r['input_port'];sig=1 if ip==1 else -1;src=np.array([[-11.5 if ip==1 else 11.5,float(al['source_center_um'][1])]]);ss=float(A.G['project_points'](src,poly)['s'][0]);s=A.G['project_points'](pv,poly)['s'];db=bet[f'straight_{gid}_r40']['beta_FDTD']-bet[f'straight_{gid}_r30']['beta_FDTD'];weights=[abs(complex(z))**2 for z in hi];raw=[];res=[]
 for u,v,sv in zip(lo,hi,s):
  ph=cmath.phase(complex(v)*complex(u).conjugate());raw.append(abs(ph));res.append(abs(cmath.phase(cmath.exp(1j*(ph-db*sig*(float(sv)-ss))))))
 norm=sum(weights);rw=sum(w*v for w,v in zip(weights,raw))/norm*180/math.pi;rr=sum(w*v for w,v in zip(weights,res))/norm*180/math.pi;expected=next(v for v in read(R/'decomposition.json') if v['geometry_id']==gid and v['low_resolution']==30);d=max(abs(rw-expected['bend_phase_change_wmae_deg']),abs(rr-expected['residual_after_delta_beta_wmae_deg']));assert d<1e-10;checks.append(dict(geometry_id=gid,input_port=ip,raw_wmae_deg=rw,residual_wmae_deg=rr,max_scalar_difference_deg=d))
save(R/'independent_decomposition_verification.json',dict(passed=True,checks=checks,no_phase_or_scale_fit=True));print('CIRCULAR_SCALAR_VERIFIED',flush=True)
