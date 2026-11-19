"""Explicit local failures from fixed cached diagnostics; no changes to carriers."""
import csv
import numpy as np
from common import OUT,OLD,read,save
from path_geometry import points_on_arc,curve_info
def main():
    rows=[];tolerance=1e-10
    for rec in read(OLD/'sbend_index.json'):
        gid=rec['geometry_id'];w=rec['params']['wg_width_um']
        with np.load(OUT/'profiles'/f'{gid}.npz') as d,np.load(OUT/'maps'/f'{gid}.npz') as maps:
            points=d['points_um'];E=d['Ez_samples'];gate=d['amplitude_gate'];ns=d['n_um'];center=np.argmin(abs(ns));bend=maps['bend_um']
            nm=np.broadcast_to(ns,points.shape[:2]);xx=points[...,0]
            # Signed curvature at curve-sampling locations, used only to name inside/outside of each bend.
            poly=maps['polyline_um'];ss=d['s_um'];_,tt=points_on_arc(ss,poly)
            theta=np.unwrap(np.arctan2(tt[:,1],tt[:,0]));curvature=np.gradient(theta,ss)
            kmesh=np.broadcast_to(curvature[:,None],nm.shape)
            zones={'core_all':(abs(nm)<=w/2+1e-10)&(xx>=-11)&(xx<=11),
                   'core_bend':(abs(nm)<=w/2+1e-10)&(xx>=bend[0,0])&(xx<=bend[-1,0]),
                   'inner_half':(abs(nm)<=w/2+1e-10)&(nm*kmesh>0)&(abs(kmesh)>.02),
                   'outer_half':(abs(nm)<=w/2+1e-10)&(nm*kmesh<0)&(abs(kmesh)>.02),
                   'core_leads':(abs(nm)<=w/2+1e-10)&((xx<bend[0,0])|(xx>bend[-1,0]))&(xx>=-11)&(xx<=11),
                   'halo':(abs(nm)>w/2+1e-10)&(abs(nm)<=w/2+.5)&(xx>=-11)&(xx<=11)}
            for axis,ax,delta in [('propagation',0,.05),('normal',1,.025)]:
                aa=[slice(None)]*2;bb=aa.copy();aa[ax]=slice(None,-1);bb[ax]=slice(1,None);a=tuple(aa);b=tuple(bb)
                gradients={name:np.angle(E[b]*E[a].conj()*np.exp(-1j*(d[name+'_phase'][b]-d[name+'_phase'][a])))/delta for name in ['linear','column_arc','local_tangent','path']}
                wt=abs(E[a])*abs(E[b])
                for zone,mask in zones.items():
                    valid=mask[a]&mask[b]&gate[a]&gate[b]&np.isfinite(wt)
                    for baseline in ['linear','column_arc','local_tangent']:
                        g0=gradients[baseline];g1=gradients['path'];dif=abs(g1)-abs(g0);n=int(valid.sum())
                        if not n:continue
                        fail=valid&(dif>tolerance)
                        weighted_fail=float(wt[fail].sum()/wt[valid].sum())
                        indices=np.argwhere(valid);j=np.argmax(dif[valid]);worst=tuple(indices[j]);pos=.5*(points[a][worst]+points[b][worst])
                        rows.append({'geometry_id':gid,'input_port':rec['input_port'],'R_min_um':rec['params']['R_min_um'],'axis':axis,'zone':zone,'baseline':baseline,'pairs':n,
                            'path_worse_fraction':float(fail.sum()/n),'path_worse_weight_fraction':weighted_fail,
                            'baseline_weighted_rms':float(np.sqrt(np.sum(wt[valid]*g0[valid]**2)/wt[valid].sum())),
                            'path_weighted_rms':float(np.sqrt(np.sum(wt[valid]*g1[valid]**2)/wt[valid].sum())),
                            'max_local_increase_rad_um':float(dif[valid].max()),'worst_x_um':float(pos[0]),'worst_y_um':float(pos[1])})
    with (OUT/'local_failure_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    save('local_failure_cases.json',rows)
    print('Saved pointwise and inner/outer failure audit; carrier definitions unchanged.')
if __name__=='__main__':main()
