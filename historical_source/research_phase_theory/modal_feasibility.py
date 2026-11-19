"""Independent CLOSED scalar constant-curvature eigenproblem, not a carrier."""
from analyze_phase import *
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

def slab_beta(width,ec,eb):
 k0=2*np.pi/1.55;a=width/2;V=a*k0*np.sqrt(ec-eb)
 u=brentq(lambda u:u*np.tan(u)-np.sqrt(max(V*V-u*u,0)),1e-10,min(V,np.pi/2)-1e-10)
 return float(np.sqrt(ec*k0*k0-(u/a)**2))

def solve(width,ec,eb,kappa,half,step):
 count=int(round(2*half/step));n=np.linspace(-half,half,count+1)[1:-1];h=1-kappa*n
 assert h.min()>0
 faces=1-kappa*(np.r_[n[0]-step/2,n+step/2])
 # Cell-averaged core fill fraction avoids moving staircase interface bias.
 fill=np.maximum(0,np.minimum(n+step/2,width/2)-np.maximum(n-step/2,-width/2))/step
 eps=eb+(ec-eb)*fill;k0=2*np.pi/1.55
 diag=-(faces[:-1]+faces[1:])*h/step**2+eps*h*h*k0*k0
 off=faces[1:-1]*np.sqrt(h[:-1]*h[1:])/step**2
 vals,vec=eigh_tridiagonal(diag,off,select='i',select_range=(len(n)-12,len(n)-1))
 beta0=slab_beta(width,ec,eb);a=width/2;q=np.sqrt(ec*k0*k0-beta0*beta0);gamma=np.sqrt(beta0*beta0-eb*k0*k0)
 ref=np.where(abs(n)<=a,np.cos(q*n),np.cos(q*a)*np.exp(-gamma*(abs(n)-a)));ref/=np.linalg.norm(ref)
 psi=np.sqrt(h[:,None])*vec;psi/=np.linalg.norm(psi,axis=0)
 overlap=abs(ref@psi)**2;j=int(overlap.argmax());v=psi[:,j]
 return dict(beta_rad_um=float(np.sqrt(vals[j])),overlap_with_straight=float(overlap[j]),
  core_energy_fraction=float(np.sum(abs(v)**2*fill)),transverse_centroid_um=float(np.sum(abs(v)**2*n)),
  selected_descending_branch=int(len(vals)-1-j),min_h=float(h.min()),analytic_straight_beta=beta0)

def main():
 records=[r for r in dataset() if r['subset']=='development16'];materials={}
 for r in records:
  with np.load(r['path'],allow_pickle=False) as z: ec=float(z['eps'].max());eb=float(z['eps'].min())
  materials[r['params']['wg_width_um']]=(ec,eb)
 rows=[]
 for width,(ec,eb) in sorted(materials.items()):
  for box in CONFIG['modal_box_halfwidths_um']:
   for step in CONFIG['modal_mesh_um']:
    zero=solve(width,ec,eb,0,box,step)['beta_rad_um']
    for kap in CONFIG['modal_kappa_per_um']:
     a=solve(width,ec,eb,kap,box,step)
     rows.append(dict(width_um=width,eps_core=ec,eps_clad=eb,kappa_per_um=kap,box_halfwidth_um=box,step_um=step,
       **a,delta_beta_rad_um=a['beta_rad_um']-zero,c2_diagnostic_um=(a['beta_rad_um']-zero)/kap**2 if kap else None))
 csvout('closed_scalar_modal_grid.csv',rows)
 summaries=[];phase=[]
 for width in sorted(materials):
  fine=[r for r in rows if r['width_um']==width and r['step_um']==.005 and r['box_halfwidth_um']==2.5]
  mid=[r for r in fine if r['kappa_per_um']==.2][0];neg=[r for r in fine if r['kappa_per_um']==-.2][0]
  small=[r for r in fine if r['kappa_per_um']==.05][0]
  coarse=[r for r in rows if r['width_um']==width and r['step_um']==.01 and r['box_halfwidth_um']==2.5 and r['kappa_per_um']==.2][0]
  boxes=[r for r in rows if r['width_um']==width and r['step_um']==.005 and r['kappa_per_um']==.2]
  summaries.append(dict(width_um=width,delta_beta_k02=mid['delta_beta_rad_um'],delta_beta_k005=small['delta_beta_rad_um'],
    c2_k005=small['c2_diagnostic_um'],c2_k02=mid['c2_diagnostic_um'],mirror_beta_error=abs(mid['beta_rad_um']-neg['beta_rad_um']),
    mesh_beta_difference=abs(mid['beta_rad_um']-coarse['beta_rad_um']),box_delta_beta_range=float(np.ptp([r['delta_beta_rad_um'] for r in boxes])),
    straight_analytic_relative_error=abs(small['analytic_straight_beta']-[r for r in fine if r['kappa_per_um']==0][0]['beta_rad_um'])/small['analytic_straight_beta'],
    min_overlap=min(r['overlap_with_straight'] for r in fine),closed_boundary_not_certified=True))
  # Scale estimate on fixed geometry; NOT a phase carrier, no prediction/learning.
  for r in records:
   if r['params']['wg_width_um']!=width:continue
   with np.load(OUT/'dev_geometry'/f"{r['geometry_id']}.npz") as z:poly=z['polyline_um']
   _,_,arc,tangent=V1.curve_info(poly);ss=np.arange(0,arc[-1],.05)
   angle=np.interp(ss,(arc[:-1]+arc[1:])/2,np.unwrap(np.arctan2(tangent[:,1],tangent[:,0])))
   kap=np.gradient(gaussian_filter1d(angle,2.),.05)
   phase.append(dict(geometry_id=r['geometry_id'],input_port=r['input_port'],width_um=width,
     integral_kappa_squared_per_um=float(np.trapezoid(kap**2,ss)),max_kappa_per_um=float(abs(kap).max()),
     small_curvature_c2_um=small['c2_diagnostic_um'],closed_model_phase_scale_rad=float(small['c2_diagnostic_um']*np.trapezoid(kap**2,ss)),
     observed_kappa_exceeds_tabulation=bool(abs(kap).max()>.25)))
 csvout('closed_scalar_modal_summary.csv',summaries);csvout('closed_scalar_phase_scale.csv',phase)
 write('modal_scope.json',{'problem':'(h psi prime) prime + k0^2 eps h psi = beta^2 psi/h; h=1-kappa*n',
  'boundary':'psi(-a)=psi(a)=0; real self-adjoint closed box; no outgoing radiation',
  'branch':'highest L2 overlap with analytic straight fundamental among top 12 closed eigenvectors',
  'not_computed':['open bent resonant modes','radiation loss','Maxwell full-vector modes','new formal carrier'],
  'rows':len(rows),'materials':len(materials),'script_sha256':sha(__file__)})
 print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
