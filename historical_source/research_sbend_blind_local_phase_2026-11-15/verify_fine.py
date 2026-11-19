"""Independent fine-grid coordinate/interpolation audit, saved arrays only."""
import math,cmath
import numpy as np
from experiment import R,read,save
from evaluate_blind import RECORDS,A,load,interp
cfg=read(R/'config.json');gid=cfg['fine_geometry_ids'][0];r=next(a for a in RECORDS if a['geometry_id']==gid);a0=load(R/'data'/gid/'sample.npz');poly,bend,masks=A.geometry(r,a0);pp,pm,ang=A.paircoords(r,a0,poly,bend,.5);points=np.vstack([pp[::7],pm[::7]]);checks=[]
for res in [30,40]:
 a=load(R/'fine'/gid/f'res{res}/sample.npz');f=a['Ez'];x=a['x_um'];y=a['y_um'];actual=interp(f,x,y,points);manual=[]
 for pt in points:
  u=(pt[0]-x[0])/(x[1]-x[0]);v=(pt[1]-y[0])/(y[1]-y[0]);j=math.floor(u);i=math.floor(v);du=u-j;dv=v-i;manual.append((1-du)*(1-dv)*complex(f[i,j])+du*(1-dv)*complex(f[i,j+1])+(1-du)*dv*complex(f[i+1,j])+du*dv*complex(f[i+1,j+1]))
 diff=float(max(abs(actual-np.asarray(manual))));assert diff<1e-10
 assert np.max(abs(a['source_center_um']-a0['source_center_um']))<1e-12
 assert abs(x[0]-(-13+.5/res))<1e-10 and abs(y[0]-(-5+.5/res))<1e-10
 checks.append(dict(resolution=res,points=len(points),max_hand_bilinear_complex_difference=diff,x_first=float(x[0]),y_first=float(y[0]),source_physical_center_identical=True,shape=list(f.shape)))
save(R/'fine_coordinate_verification.json',dict(passed=True,geometry_id=gid,checks=checks,wrapped_phase_interpolated=False));print('FINE_COORDINATE_VERIFIED',checks,flush=True)
