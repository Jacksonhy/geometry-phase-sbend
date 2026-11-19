"""Saved-array checks independent of vector metric implementation."""
import sys,cmath,math
import numpy as np
from experiment import R,read,save,sha,SEEDS,METHODS
from evaluate_blind import A,RECORDS,load,predictions
P=predictions();max_pair=0.;max_rel=0.;count=0;arc_checks=[]
# Hand bilinear interpolation: separate scalar indices and four-neighbor weights.
def scalar(f,x,y,pt):
 u=(float(pt[0])-float(x[0]))/float(x[1]-x[0]);v=(float(pt[1])-float(y[0]))/float(y[1]-y[0]);j=math.floor(u);i=math.floor(v);du=u-j;dv=v-i
 assert 0<=i<f.shape[0]-1 and 0<=j<f.shape[1]-1
 return complex((1-du)*(1-dv)*f[i,j]+du*(1-dv)*f[i,j+1]+(1-du)*dv*f[i+1,j]+du*dv*f[i+1,j+1])
for index in [0,35,71]:
 r=RECORDS[index];a=load(R/'data'/r['geometry_id']/'sample.npz');e=a['Ez'];poly,bend,masks=A.geometry(r,a);ma=float(abs(e[masks['bend_core']]).max());plus,minus,ang=A.paircoords(r,a,poly,bend,.5)
 for s in SEEDS:
  for m in METHODS:
   p=P[s,m][index].astype(np.complex128);true=[];pred=[]
   for qp,qm in zip(plus,minus):
    ep=scalar(e,a['x_um'],a['y_um'],qp);em=scalar(e,a['x_um'],a['y_um'],qm)
    if abs(ep)<.05*ma or abs(em)<.05*ma:continue
    pp=scalar(p,a['x_um'],a['y_um'],qp);pm=scalar(p,a['x_um'],a['y_um'],qm);true.append(ep*em.conjugate());pred.append(pp*pm.conjugate())
   weights=[abs(z) for z in true];pv=sum(w*abs(cmath.phase(cp/ct)) if abs(cp)>1e-24*ma**2 else w*math.pi for w,cp,ct in zip(weights,pred,true))/sum(weights)*180/math.pi;rv=math.sqrt(sum(abs(cp-ct)**2 for cp,ct in zip(pred,true))/sum(abs(ct)**2 for ct in true));ref=A.pairmetric(p,e,a,plus,minus,ang,.05,ma,None);max_pair=max(max_pair,abs(pv-ref['normal_relative_phase_wmae_deg']));max_rel=max(max_rel,abs(rv-ref['coherence_complex_rel_l2']));count+=1
 # Independently scalar-integrate arc carrier including source direction.
 with np.load(R/'cache'/(r['geometry_id']+'.npz')) as z:c=z['carrier_arc_only'][0]
 from experiment import OLD
 cal=read(OLD/'calibration.json');beta=float(np.polyval(cal['coefficients_rad_um'],a['eps'].max()));source=a['source_center_um'];sig=1 if r['input_port']==1 else -1
 def arc_at(x):
  total=0.
  for p,q in zip(poly[:-1],poly[1:]):
   if x>=q[0]:total+=float(np.linalg.norm(q-p))
   elif x>p[0]:total+=float(np.linalg.norm(q-p))*(x-p[0])/(q[0]-p[0]);break
   else:break
  return total
 errors=[]
 for j in [0,150,300,479]:errors.append(abs(c[80,j]-cmath.exp(1j*beta*sig*(arc_at(a['x_um'][j])-arc_at(source[0])))))
 assert max(errors)<2e-7;arc_checks.append(dict(geometry_id=r['geometry_id'],input_port=r['input_port'],max_complex_difference=float(max(errors))))
assert max_pair<1e-9 and max_rel<1e-10
save(R/'independent_verification.json',dict(passed=True,normal_pair_hand_bilinear_checks=count,max_phase_difference_deg=max_pair,max_cross_term_relL2_difference=max_rel,arc_carrier_scalar_integral=arc_checks))
print('INDEPENDENT_CHECKS_PASSED',max_pair,max_rel,flush=True)


