"""Independent scalar check of longitudinal beta and transmission phase closure."""
import sys,math,cmath
import numpy as np
from experiment import R,read,save
from analyze import load,interpolate,beta_one
checks=[]
for name in ['centered_w0.400_r20','centered_w0.500_r40','centered_w0.575_r40','straight_blindB_cce44732dceff594_r40','straight_blindB_9f78121d6c81e7b2_r40']:
 p=R/'runs'/name
 if not (p/'result.json').exists():continue
 a=load(p/'sample.npz');x=a['x_um'];cy=float(a['source_center_um'][1]);line=interpolate(a,np.column_stack([x,np.full(len(x),cy)]));mask=(x>=-9)&(x<=9);sig=1 if int(a['input_port'])==1 else -1;xx=sig*x[mask];vv=line[mask];phase=[];last=None
 for z in vv:
  raw=cmath.phase(complex(z))
  if last is None:q=raw
  else:q=raw+2*math.pi*round((last-raw)/(2*math.pi))
  phase.append(q);last=q
 w=[abs(complex(z))**2 for z in vv];sw=sum(w);xm=sum(float(v)*q for v,q in zip(xx,w))/sw;pm=sum(v*q for v,q in zip(phase,w))/sw;num=sum(q*(float(xv)-xm)*(pv-pm) for q,xv,pv in zip(w,xx,phase));den=sum(q*(float(xv)-xm)**2 for q,xv in zip(w,xx));b=num/den;vec=beta_one(p);assert abs(b-vec['beta_centerline'])<1e-10
 t=complex(vec['physical_t_real'],vec['physical_t_imag']);closure=math.degrees(abs(cmath.phase(t*cmath.exp(-1j*vec['beta_FDTD']*23))))
 checks.append(dict(job=name,scalar_beta=b,vector_beta=vec['beta_centerline'],difference=abs(b-vec['beta_centerline']),physical_transmission_vs_beta23_phase_residual_deg=closure))
save(R/'independent_beta_verification.json',dict(passed=True,checks=checks,phase_unwrap_scalar=True,weighted_regression_scalar=True));print('SCALAR_BETA_CHECKS',checks,flush=True)

