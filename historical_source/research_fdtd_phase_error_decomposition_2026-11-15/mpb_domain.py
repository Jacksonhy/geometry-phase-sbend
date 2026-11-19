"""MPB transverse-domain sensitivity, no time-domain propagation."""
import numpy as np,time
import experiment as e
from analyze import csvsave
out=[]
for w in e.read(e.R/'config.json')['widths_um']:
 j=dict(id=f'mpb_domain_w{w:.3f}',kind='centered',width=w,resolution=60);fw,r,dev,sim,mons,dft,dh=e.build(j);mp=fw.mp;sim.init_sim()
 for span in [w+1,w+2,w+4]:
  vol=mp.Volume(center=mp.Vector3(-11.5,0),size=mp.Vector3(0,span));md=sim.get_eigenmode(1/1.55,mp.X,vol,1,mp.Vector3(),resolution=240,parity=mp.NO_PARITY);out.append(dict(width_um=w,FDTD_geometry_resolution=60,MPB_eig_resolution=240,transverse_span_um=span,beta_rad_um=float(2*np.pi*abs(md.k.x))))
 sim.reset_meep();print('MPB_DOMAIN',w,flush=True)
e.save(e.R/'mpb_transverse_domain.json',out);csvsave(e.R/'mpb_transverse_domain.csv',out)
