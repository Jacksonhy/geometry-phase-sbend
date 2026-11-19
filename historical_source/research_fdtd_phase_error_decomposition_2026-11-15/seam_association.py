"""Diagnostic transverse cross-term changes and seam-location association."""
import numpy as np
from experiment import R,read,save,prior
from analyze import RECORDS,A,sbend,load,interpolate,csvsave
rows=[]
for r in RECORDS:
 gid=r['geometry_id'];a0=load(prior(gid,20));poly,bend,masks=A.geometry(r,a0);plus,minus,ang=A.paircoords(r,a0,poly,bend,.5);xx=(plus[:,0]+minus[:,0])/2;yy=(plus[:,1]+minus[:,1])/2;centers=np.column_stack([xx,yy]);vtx=bend[1:-1];dist=np.sqrt(((centers[:,None]-vtx[None])**2).sum(-1)).min(axis=1)
 for lo,hi in [(20,30),(30,40),(40,60)]:
  al=sbend(gid,lo);ah=sbend(gid,hi);lp=interpolate(al,plus);lm=interpolate(al,minus);hp=interpolate(ah,plus);hm=interpolate(ah,minus);phase=abs(np.angle((hp*hm.conj())*(lp*lm.conj()).conj()));weights=abs(hp*hm.conj());cut=float(np.median(dist));X,Y=np.meshgrid(a0['x_um'],a0['y_um']);ma=abs(interpolate(ah,np.stack([X,Y],axis=-1))[masks['bend_core']]).max();gate=(abs(hp)>=.05*ma)&(abs(hm)>=.05*ma)
  for name,m in [('nearer_vertex_half',dist<=cut),('farther_vertex_half',dist>cut)]:
   good=gate&m;rows.append(dict(geometry_id=gid,width_um=r['params']['wg_width_um'],angle_stratum=r['angle_stratum'],low_resolution=lo,high_resolution=hi,group=name,points=int(good.sum()),median_split_vertex_distance_um=cut,normal_phase_wmae_deg=float(np.degrees(np.sum(weights[good]*phase[good])/weights[good].sum()))))
save(R/'seam_location_association.json',dict(descriptive_only=True,causal_attribution=False,limitation='Vertices occur densely along every original Block join; distance strongly coupled to local x sampling, angle and resolution; this is not isolated seam-only perturbation',rows=rows));csvsave(R/'seam_location_association.csv',rows);print('SEAM_ASSOCIATION_SAVED',flush=True)
