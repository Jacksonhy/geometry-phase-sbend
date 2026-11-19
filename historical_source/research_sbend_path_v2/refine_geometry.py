"""Independent fixed-sigma grid refinement, not a parameter/candidate search."""
import time
import numpy as np
from support import OUT,V1,OLD,read,save,csvsave
import path_v2 as v2

def main():
    rows=[];details=[]
    for rec in read(OLD/'sbend_index.json'):
        gid=rec['geometry_id'];ip=rec['input_port'];w=rec['params']['wg_width_um'];t0=time.monotonic()
        with np.load(V1/'maps'/f'{gid}.npz') as m:
            poly=m['polyline_um'];src=m['source_center_um'];beta=float(m['beta_Y_rad_um'])
            for h in [.05,.025]:
                # Same interior physical support; algorithm sigma remains .05.
                x=np.arange(-11.975,11.975+h/4,h);y=np.arange(-3.975,3.975+h/4,h)
                pts=np.stack(np.meshgrid(x,y),axis=-1)
                f=v2.distances_at_points(pts,poly,src,ip,w,.05)
                D=f['path_v2']
                for axis,ax in [('x',1),('y',0)]:
                    delta=np.diff(D,axis=ax);j=np.unravel_index(abs(delta).argmax(),delta.shape)
                    rows.append({'geometry_id':gid,'grid_step_um':h,'fixed_sigma_um':.05,'axis':axis,
                        'max_abs_distance_gradient':float(abs(delta[j])/h),
                        'max_abs_unwrapped_phase_increment_rad':float(abs(beta*delta[j])),
                        'max_complex_carrier_difference':float(abs(np.exp(1j*beta*delta)-1).max()),
                        'nonfinite_count':int((~np.isfinite(D)).sum())})
            # Refinement at independently found v2 maximum local grid edge.
            old=[r for r in read(OUT/'gradient_checks.json') if r['geometry_id']==gid and r['method']=='path_v2' and r['band']=='all']
            worst=max(old,key=lambda r:r['max_abs_coordinate_gradient'])
            center=np.array([worst['x_um'],worst['y_um']]);axis=np.array([1.,0.]) if worst['axis']=='x' else np.array([0.,1.])
            for e in [1e-2,1e-4,1e-6,1e-8]:
                p=np.stack([center-axis*e,center+axis*e])
                d=v2.distances_at_points(p,poly,src,ip,w,.05)['path_v2']
                details.append({'geometry_id':gid,'epsilon_um':e,'x_um':center[0],'y_um':center[1],
                    'distance_difference_um':float(abs(d[1]-d[0])),'gradient':float(abs(d[1]-d[0])/(2*e))})
        print(f'refined {gid} {time.monotonic()-t0:.2f}s',flush=True)
    csvsave('grid_refinement.csv',rows);save('grid_refinement.json',rows)
    csvsave('v2_peak_refinement.csv',details);save('v2_peak_refinement.json',details)
    save('refinement_summary.json',{'sigma_fixed_um':.05,'candidate_count':1,
        'grid_max_gradient':{str(h):max(r['max_abs_distance_gradient'] for r in rows if r['grid_step_um']==h) for h in [.05,.025]},
        'peak_epsilon_1e8_max_gradient':max(r['gradient'] for r in details if r['epsilon_um']==1e-8),
        'nonfinite_count':sum(r['nonfinite_count'] for r in rows),'parameters_changed':False})
    print(read(OUT/'refinement_summary.json'),flush=True)
if __name__=='__main__':main()
