"""Physical-distance and far-cladding checks, supplementing unchanged s-coordinate metrics."""
import csv
import numpy as np
from common import OUT,OLD,read,save
def main():
    physical=[];far=[]
    for rec in read(OLD/'sbend_index.json'):
        gid=rec['geometry_id'];w=rec['params']['wg_width_um']
        with np.load(OUT/'maps'/f'{gid}.npz') as m,np.load(OUT/'profiles'/f'{gid}.npz') as d:
            P=d['points_um'];E=d['Ez_samples'];g=d['amplitude_gate'];n=d['n_um'];bend=m['bend_um']
            mid=.5*(P[:-1]+P[1:]);ds=np.linalg.norm(np.diff(P,axis=0),axis=-1)
            valid=(abs(n[None,:])<=w/2+1e-10)&(P[:-1,:,0]>=-11)&(P[1:,:,0]<=11)&g[:-1]&g[1:]&(ds>1e-9)
            weights=abs(E[:-1])*abs(E[1:])
            for name in ['linear','column_arc','local_tangent','path']:
                gg=np.angle(E[1:]*E[:-1].conj()*np.exp(-1j*np.diff(d[name+'_phase'],axis=0)))/ds
                for zone,mask in [('all',np.ones_like(valid)),('bend',(mid[...,0]>=bend[0,0])&(mid[...,0]<=bend[-1,0]))]:
                    v=valid&mask&np.isfinite(gg)
                    physical.append({'geometry_id':gid,'input_port':rec['input_port'],'method':name,'zone':zone,
                        'physical_tangent_weighted_rms':float(np.sqrt(np.sum(weights[v]*gg[v]**2)/weights[v].sum())),
                        'physical_step_min_um':float(ds[v].min()),'physical_step_max_um':float(ds[v].max())})
            x=m['x_um'];y=m['y_um'];N=m['projection_distance_um']
            for name in ['linear','column_arc','local_tangent','path']:
                D=m[name+'_um']
                for axis,ax in [('x',1),('y',0)]:
                    diff=np.diff(D,axis=ax);grad=diff/.05
                    j=np.unravel_index(np.argmax(abs(grad)),grad.shape)
                    k=list(j);k[ax]+=1;k=tuple(k)
                    distance=float(.5*(N[j]+N[k]));phasediff=m['beta_Y_rad_um']*diff[j]
                    far.append({'geometry_id':gid,'input_port':rec['input_port'],'method':name,'axis':axis,
                        'max_abs_coordinate_gradient':float(abs(grad[j])),'x_um':float((x[j[1]]+x[k[1]])/2),
                        'y_um':float((y[j[0]]+y[k[0]])/2),'distance_to_centerline_um':distance,
                        'coordinate_jump_um':float(diff[j]),'unwrapped_carrier_phase_jump_rad':float(phasediff),
                        'wrapped_carrier_phase_jump_rad':float(np.angle(np.exp(1j*phasediff))),
                        'gradient_max_core':None})
                # All core-neighbor edges and global edges over fixed threshold10.
                for axis,ax in [('x',1),('y',0)]:
                    aa=[slice(None)]*2;bb=aa.copy();aa[ax]=slice(None,-1);bb[ax]=slice(1,None);a=tuple(aa);b=tuple(bb)
                    grad=abs((D[b]-D[a])/.05);c=(N[a]<=w/2)&(N[b]<=w/2)
                    far[-2 if ax==1 else -1]['gradient_max_core']=float(grad[c].max())
                    far[-2 if ax==1 else -1]['edge_count_gradient_gt_10']=int((grad>10).sum())
        print(f'physical and far-field check {gid}',flush=True)
    cal=read(OUT/'calibration_transfer.json')
    fields=np.array([abs(r['relative_error_Y_vs_field']) for r in cal]);bias=np.array([abs((r['beta_Y_rad_um']-r['beta_downstream_field_rad_um'])*r['path_to_output_monitor_um']) for r in cal])
    save('calibration_field_crosscheck.json',{'query_beta_used_in_carrier':False,'new_coefficients_fitted':False,'FDTD_lead_relative_MAE':float(fields.mean()),
        'FDTD_lead_relative_max':float(fields.max()),'FDTD_lead_accumulated_phase_max':float(bias.max()),
        'same_numerical_thresholds_passed':bool(fields.mean()<=.01 and fields.max()<=.02 and bias.max()<=np.pi/4),
        'interpretation':'MPB wavevector and spatial DFT phase slope are distinct references on this grid; MPB gate failure alone is not proof of field mismatch. Official training will fit new training fields only.'})
    save('physical_tangent_diagnostics.json',physical);save('far_field_extension.json',far)
    for name,rows in [('physical_tangent_diagnostics.csv',physical),('far_field_extension.csv',far)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
if __name__=='__main__':main()
