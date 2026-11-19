"""Original field and normal-pair metrics, with portable geometry imports."""
import numpy as np
from scipy.ndimage import map_coordinates
from . import geometry as geom
G={k:getattr(geom,k) for k in ["centerline_from_params","project_points","smooth_tangent_at_x"]}

def geometry(rec,a):
 x,y=a['x_um'],a['y_um'];poly,bend=G['centerline_from_params'](rec['params']);X,Y=np.meshgrid(x,y);projection=G['project_points'](np.stack([X,Y],axis=-1),poly)
 p=projection['closest'];tx=G['smooth_tangent_at_x'](p[...,0],poly);angle=np.degrees(np.arctan2(abs(tx[...,1]),tx[...,0]));eta=2*projection['normal_offset']/rec['params']['wg_width_um'];is_bend=(X>=bend[0,0])&(X<=bend[-1,0]);core=projection['distance']<=rec['params']['wg_width_um']/2
 masks={'whole':np.ones(core.shape,bool),'bend_core':is_bend&core,'straight_core':~is_bend&core,'bend_near_cladding':is_bend&(abs(eta)>1)&(abs(eta)<=3)}
 ab=[0.,10.,20.,35.,90.];eb=[-3.,-1.,-.5,0.,.5,1.,3.]
 for k in range(len(ab)-1):
  am=(angle>=ab[k])&(angle<ab[k+1]);masks[f'angle{k}_core']=is_bend&core&am
  for j in range(len(eb)-1):masks[f'cell_a{k}_n{j}']=is_bend&am&(eta>=eb[j])&(eta<eb[j+1])
 for j in range(len(eb)-1):masks[f'normal{j}']=is_bend&(eta>=eb[j])&(eta<eb[j+1])
 return poly,bend,masks

def fieldmetric(p,e,mask,gate,maxamp):
 n=int(mask.sum())
 if n<10:return None
 power=abs(e)**2;den=float(power[mask].sum())
 if den<=0:return None
 err=abs(p-e)**2;delta=np.angle(p*e.conj());delta=np.where(abs(p)>1e-12*maxamp,delta,np.pi)
 good=mask&(abs(e)>=gate*maxamp);pd=float(power[good].sum());ng=int(good.sum())
 if ng<10 or pd<=0:return None
 amp=abs(p)-abs(e);phasepart=2*abs(p)*abs(e)*(1-np.cos(delta));decomp=float((amp[mask]**2+phasepart[mask]).sum())
 assert abs(decomp-float(err[mask].sum()))<1e-8*max(den,1.)
 return dict(pixels=n,phase_pixels=ng,true_energy_fraction=float(den/power.sum()),complex_rel_l2=float(np.sqrt(err[mask].sum()/den)),phase_wmae_deg=float(np.degrees(np.sum(power[good]*abs(delta[good]))/pd)),phase_phasor_l2=float(np.sqrt(np.sum(power[good]*abs(np.exp(1j*delta[good])-1)**2)/pd)),amplitude_rel_l2=float(np.sqrt(np.sum(amp[mask]**2)/den)),phase_term_normalized_mse=float(np.sum(phasepart[mask])/den))

def samples(field,x,y,points):
 coords=np.stack([(points[:,1]-y[0])/(y[1]-y[0]),(points[:,0]-x[0])/(x[1]-x[0])]);return map_coordinates(field,coords,order=1,mode='nearest',prefilter=False)

def paircoords(rec,a,poly,bend,eta):
 x=a['x_um'];xx=x[(x>=bend[0,0]+.1)&(x<=bend[-1,0]-.1)];yc=np.interp(xx,poly[:,0],poly[:,1]);t=G['smooth_tangent_at_x'](xx,poly);normal=np.column_stack([-t[:,1],t[:,0]]);base=np.column_stack([xx,yc]);distance=eta*rec['params']['wg_width_um']/2
 return base+distance*normal,base-distance*normal,np.degrees(np.arctan2(abs(t[:,1]),t[:,0]))

def pairmetric(p,e,a,plus,minus,angle,gate,maxamp,anglebin):
 ep=samples(e,a['x_um'],a['y_um'],plus);em=samples(e,a['x_um'],a['y_um'],minus);pp=samples(p,a['x_um'],a['y_um'],plus);pm=samples(p,a['x_um'],a['y_um'],minus)
 good=(abs(ep)>=gate*maxamp)&(abs(em)>=gate*maxamp)
 if anglebin is not None:
  bounds=[0,10,20,35,90];good&=(angle>=bounds[anglebin])&(angle<bounds[anglebin+1])
 if good.sum()<8:return None
 ct=ep[good]*em[good].conj();cp=pp[good]*pm[good].conj();phase=np.angle(cp*ct.conj());phase=np.where(abs(cp)>1e-24*maxamp**2,phase,np.pi);w=abs(ct)
 return dict(pairs=int(good.sum()),normal_relative_phase_wmae_deg=float(np.degrees(np.sum(w*abs(phase))/w.sum())),coherence_complex_rel_l2=float(np.sqrt(np.sum(abs(cp-ct)**2)/np.sum(abs(ct)**2))),normal_relative_phase_phasor_l2=float(np.sqrt(np.sum(w*abs(np.exp(1j*phase)-1)**2)/w.sum())))
