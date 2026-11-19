"""Gauge-invariant exported transverse mode-shape comparison."""
import numpy as np
from experiment import R,read,save
from analyze import load,csvsave,RECORDS,sbend
rows=[];changes=[]
for r in RECORDS:
 gid=r['geometry_id'];w=r['params']['wg_width_um'];ip=r['input_port'];offset=np.linspace(-.5-w/2,.5+w/2,401);profiles={}
 for res in [20,30,40,60]:
  a=load(R/'runs'/f'straight_{gid}_r{res}/sample.npz');pre=f'port{ip}/';y=a[pre+'y_um']-a['source_center_um'][1];em=a[pre+'mode_Ez'];f=np.interp(offset,y,em.real)+1j*np.interp(offset,y,em.imag);norm=np.trapezoid(abs(f)**2,offset);core=abs(offset)<=w/2;fraction=np.trapezoid(abs(f[core])**2,offset[core])/norm;profiles[res]=f/np.sqrt(norm);rows.append(dict(geometry_id=gid,width_um=w,resolution=res,mode_Ez_core_energy_fraction=float(fraction),profile_interval_um=[float(offset[0]),float(offset[-1])]))
 for lo,hi in [(20,30),(30,40),(40,60)]:
  f,g=profiles[lo],profiles[hi];overlap=abs(np.trapezoid(f.conj()*g,offset));changes.append(dict(geometry_id=gid,width_um=w,low_resolution=lo,high_resolution=hi,mode_Ez_shape_overlap_squared=float(overlap**2),normalized_amplitude_shape_relL2=float(np.sqrt(np.trapezoid((abs(f)-abs(g))**2,offset)))))
save(R/'mode_shape_diagnostics.json',dict(profile_metric='Ez-only finite-window normalized shape overlap; absolute overlap removes arbitrary eigenmode gauge, not physical phase fitting or field correction',rows=rows,changes=changes));csvsave(R/'mode_shape_changes.csv',changes);print('MODE_SHAPES',len(changes),flush=True)
