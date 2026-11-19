"""Fixed development-set phase diagnostics. No query-field carrier fitting."""
import csv,json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from common import ROOT,OUT,OLD,read,save,sha,now
from path_geometry import geometry_features,curve_info,points_on_arc,distances_at_points,carriers_from_features,beta_from_training_calibration
METHODS=('linear','column_arc','local_tangent','path')

def interpolator(x,y,field):
    return RegularGridInterpolator((y,x),field,bounds_error=False,fill_value=np.nan)
def sample_at(fun,p):return fun(np.asarray(p)[...,[1,0]])
def statistics(g,w,mask):
    ok=mask&np.isfinite(g)&np.isfinite(w);v=g[ok];weights=w[ok]
    if not len(v):return {'count':0,'rms':None,'mae':None,'p95':None,'weighted_rms':None}
    return {'count':int(len(v)),'rms':float(np.sqrt(np.mean(v*v))),'mae':float(np.mean(abs(v))),
            'p95':float(np.quantile(abs(v),.95)),
            'weighted_rms':float(np.sqrt(np.sum(weights*v*v)/np.sum(weights))) if np.sum(weights)>0 else None}

def straight_slope(E,ep,x,y,poly,ip):
    """Diagnostic only: slope of real-positive scalar mode projection on downstream lead.

    Uses an average across inner half width, not S labels, no feedback into beta.
    """
    bend=poly[1:-1];w=float(ep['width']);direction=1 if ip==1 else -1
    lo,hi=(-10.5,float(bend[0,0])-.75) if ip==2 else (float(bend[-1,0])+.75,10.5)
    xs=x[(x>=lo)&(x<=hi)];yc=float(bend[0 if ip==2 else -1,1])
    ns=np.linspace(-.25*w,.25*w,5)
    points=np.stack(np.meshgrid(xs,ns,indexing='xy'),axis=-1)
    points[...,1]+=yc
    z=np.mean(sample_at(interpolator(x,y,E),points),axis=0)
    fit=np.polyfit(xs,np.unwrap(np.angle(z)),1)
    residual=np.unwrap(np.angle(z))-np.polyval(fit,xs)
    return {'beta_rad_um':float(direction*fit[0]),'slope_phase_rms':float(np.sqrt(np.mean(residual**2))),
            'count':len(xs),'x_bounds_um':[float(xs.min()),float(xs.max())],
            'role':'downstream lead field diagnostic only; never used in carriers'}

def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    t0=time.monotonic();records=read(OLD/'sbend_index.json');cal=read(ROOT/'research_e3_2400/manifest.json')['calibration']
    frozen=read(OLD/'sbend_frozen_copy.json')['records'];threshold_R=float(np.median([r['params']['R_min_um'] for r in records]))
    rows=[];geometry=[];calrows=[];clrows=[];edge_rows=[]
    for i,rec in enumerate(records):
        gid=rec['geometry_id'];folder=OLD/'sbend'/gid;result=read(folder/'result.json')
        with np.load(folder/'sample.npz',allow_pickle=False) as d:
            E=d['Ez'];eps=d['eps'].astype(float);x=d['x_um'];y=d['y_um'];src=d['source_center_um'];ip=int(d['input_port']);p=rec['params'];width=p['wg_width_um']
            # Construct carriers before reading any field-derived slope/MPB diagnostics.
            features=geometry_features(p,x,y,src,ip);poly=features['polyline_um'];bend=features['bend_um'];_,length,arc,tangent=curve_info(poly)
            beta=beta_from_training_calibration(eps.max(),.05,cal)
            carriers=carriers_from_features(features,beta)
            beta_mpb=float(np.mean([2*np.pi*m['kpoints'][0][0] for m in result['monitors']]))
            slope=straight_slope(E,{'width':width},x,y,poly,ip)
            through_length=abs(float(distances_at_points(np.array([[11.5 if ip==1 else -11.5,bend[-1 if ip==1 else 0,1]]]),poly,src,ip)['path'][0]))
            calrows.append({'geometry_id':gid,'input_port':ip,'width_um':width,'eps_max':float(eps.max()),'beta_Y_rad_um':beta,'beta_MPB_rad_um':beta_mpb,
                'beta_downstream_field_rad_um':slope['beta_rad_um'],'relative_error_Y_vs_MPB':(beta-beta_mpb)/beta_mpb,
                'relative_error_Y_vs_field':(beta-slope['beta_rad_um'])/slope['beta_rad_um'],
                'phase_bias_to_output_monitor_rad':(beta-beta_mpb)*through_length,'path_to_output_monitor_um':through_length,**{'straight_'+k:v for k,v in slope.items() if isinstance(v,(int,float))}})
            srcproj=features['source_s'];geometry.append({'geometry_id':gid,'input_port':ip,'width_um':width,'R_min_um':p['R_min_um'],'offset_requested_um':p['lateral_offset_um'],
                'actual_offset_um':float(bend[-1,1]-bend[0,1]),'bend_arc_um':float(arc[-2]-arc[1]),'bend_chord_um':float(np.linalg.norm(bend[-1]-bend[0])),
                'excess_bend_length_um':float(arc[-2]-arc[1]-(bend[-1,0]-bend[0,0])),
                'max_tangent_angle_deg':float(np.degrees(np.arctan2(tangent[:,1],tangent[:,0])).max()),
                'source_x_um':float(src[0]),'source_y_um':float(src[1]),'source_s_um':srcproj,
                'max_carrier_modulus_error':max(float(np.max(abs(abs(c)-1))) for c in carriers.values()),
                'tight_group':'smaller_R' if p['R_min_um']<=threshold_R else 'larger_R'})
            np.savez_compressed(OUT/'maps'/f'{gid}.npz',x_um=x,y_um=y,polyline_um=poly,bend_um=bend,source_center_um=src,
                beta_Y_rad_um=np.float64(beta),**{name+'_um':features[name] for name in METHODS},
                normal_offset_um=features['projection']['normal_offset'],projection_distance_um=features['projection']['distance'],
                **{name+'_carrier':c.astype(np.complex64) for name,c in carriers.items()})
            # Curvilinear frame: physical left->right s; direction appears in carrier.
            ss=np.arange(.25,arc[-1]-.249,.05);center,tt=points_on_arc(ss,poly);nn=np.stack([-tt[:,1],tt[:,0]],axis=-1)
            ns=np.arange(-1.,1.0001,.025)
            points=center[:,None,:]+ns[None,:,None]*nn[:,None,:]
            Ef=sample_at(interpolator(x,y,E),points)
            f=distances_at_points(points,poly,src,ip)
            phi={name:beta*f[name] for name in METHODS}
            center_idx=int(np.argmin(abs(ns)));Amax=float(np.max(abs(Ef[:,center_idx])));gate=abs(Ef)>=.01*Amax
            xx=points[...,0];nmesh=np.broadcast_to(ns[None,:],xx.shape)
            zone={'all':(xx>=-11)&(xx<=11),'bend':(xx>=bend[0,0])&(xx<=bend[-1,0]),
                  'leads':((xx<bend[0,0])|(xx>bend[-1,0]))&(xx>=-11)&(xx<=11)}
            bands={'core':abs(nmesh)<=width/2+1e-10,'center':abs(nmesh)<=.25*width+1e-10,
                   'negative_side':(nmesh>=-width/2-1e-10)&(nmesh<-.25*width),
                   'positive_side':(nmesh<=width/2+1e-10)&(nmesh>.25*width),
                   'halo':(abs(nmesh)>width/2+1e-10)&(abs(nmesh)<=width/2+.5+1e-10)}
            for method in METHODS:
                for axis,ax,delta in [('propagation',0,.05),('normal',1,.025)]:
                    pair0=[slice(None)]*2;pair1=pair0.copy();pair0[ax]=slice(None,-1);pair1[ax]=slice(1,None);a=tuple(pair0);b=tuple(pair1)
                    g=np.angle(Ef[b]*Ef[a].conj()*np.exp(-1j*(phi[method][b]-phi[method][a])))/delta
                    wt=abs(Ef[a])*abs(Ef[b]);gpair=gate[a]&gate[b]
                    for zn,zm in zone.items():
                        for bn,bm in bands.items():
                            region=zm[a]&zm[b]&bm[a]&bm[b]
                            for gating in ('amplitude_1pct','ungated'):
                                stats=statistics(g,wt,region&(gpair if gating=='amplitude_1pct' else True))
                                rows.append({'geometry_id':gid,'input_port':ip,'R_group':geometry[-1]['tight_group'],'method':method,'frame':'curvilinear',
                                    'axis':axis,'zone':zn,'band':bn,'gate':gating,**stats})
            centerz=Ef[:,center_idx];pc=distances_at_points(center,poly,src,ip)
            select=(center[:,0]>=-11)&(center[:,0]<=11);phase=np.unwrap(np.angle(centerz))
            ss0=ss[select];phase=phase[select];source_direction=1 if ip==1 else -1
            for method in METHODS:
                residual=phase-beta*pc[method][select]
                increments=np.diff(residual);g=source_direction*increments/.05
                clrows.append({'geometry_id':gid,'input_port':ip,'method':method,'unwrapped_net_residual_rad':float(source_direction*(residual[-1]-residual[0])),
                    'residual_total_variation_rad':float(np.sum(abs(increments))),'residual_gradient_rms':float(np.sqrt(np.mean(g*g))),
                    'phase_range_rad':float(np.ptp(residual))})
            # Cartesian gradients for a separate check on complete original grid.
            X,Y=np.meshgrid(x,y);core=eps>(eps.min()+eps.max())/2
            cartgate=abs(E)>=.01*Amax;central=(X>=-11)&(X<=11)
            for method in METHODS:
                ph=beta*features[method]
                for axis,ax,delta in [('x',1,.05),('y',0,.05)]:
                    a=[slice(None)]*2;b=a.copy();a[ax]=slice(None,-1);b[ax]=slice(1,None);a=tuple(a);b=tuple(b)
                    g=np.angle(E[b]*E[a].conj()*np.exp(-1j*(ph[b]-ph[a])))/delta;wt=abs(E[a])*abs(E[b])
                    for band,mask in [('core',core),('whole',np.ones_like(core))]:
                        region=mask[a]&mask[b]&central[a]&central[b]
                        for gating in ('amplitude_1pct','ungated'):
                            st=statistics(g,wt,region&((cartgate[a]&cartgate[b]) if gating=='amplitude_1pct' else True))
                            rows.append({'geometry_id':gid,'input_port':ip,'R_group':geometry[-1]['tight_group'],'method':method,'frame':'cartesian',
                                'axis':axis,'zone':'all','band':band,'gate':gating,**st})
            for method in METHODS:
                arr=features[method];gx=np.diff(arr,axis=1)/.05;gy=np.diff(arr,axis=0)/.05
                edge_rows.append({'geometry_id':gid,'method':method,'max_abs_d_distance_dx':float(abs(gx).max()),'max_abs_d_distance_dy':float(abs(gy).max()),
                    'p99_abs_d_distance_dy':float(np.quantile(abs(gy),.99))})
            # Retain fixed coordinate curves for reproducible plots; query phase never returns to constructor.
            np.savez_compressed(OUT/'profiles'/f'{gid}.npz',s_um=ss,n_um=ns,points_um=points,Ez_samples=Ef,
                **{name+'_phase':phi[name] for name in METHODS},amplitude_gate=gate)
        print(f'diagnosed {i+1}/16 {gid}',flush=True)
    csvsave('phase_diagnostics.csv',rows);csvsave('calibration_transfer.csv',calrows);csvsave('geometry_checks.csv',geometry);csvsave('centerline_residuals.csv',clrows);csvsave('extension_gradients.csv',edge_rows)
    save('geometry_checks.json',geometry);save('calibration_transfer.json',calrows);save('centerline_residuals.json',clrows)
    rel=np.array([r['relative_error_Y_vs_MPB'] for r in calrows]);bias=np.array([r['phase_bias_to_output_monitor_rad'] for r in calrows])
    calibration={'coefficients':cal,'source':'research_e3_2400/manifest.json','source_sha256':sha(ROOT/'research_e3_2400/manifest.json'),
        'beta_units':'rad/um','relative_MAE_vs_MPB':float(np.mean(abs(rel))),'relative_max_vs_MPB':float(np.max(abs(rel))),
        'max_abs_accumulated_phase_bias_rad':float(np.max(abs(bias))),
        'criteria':{'relative_MAE_max':.01,'relative_max_max':.02,'phase_bias_max':float(np.pi/4)},
        'passes_transfer_gate':bool(np.mean(abs(rel))<=.01 and max(abs(rel))<=.02 and max(abs(bias))<=np.pi/4),
        'new_beta_fitted':False,'diagnostic_mpb_or_query_slopes_used_in_carriers':False}
    save('calibration_summary.json',calibration)
    # Same geometric comparisons across all methods. No threshold selection/search.
    keys=sorted({(r['frame'],r['axis'],r['zone'],r['band'],r['gate']) for r in rows})
    summaries=[]
    for frame,axis,zone,band,gate in keys:
        for group in ('all','input1','input2','smaller_R','larger_R'):
            matching=[r for r in rows if (r['frame'],r['axis'],r['zone'],r['band'],r['gate'])==(frame,axis,zone,band,gate) and
                (group=='all' or group==f"input{r['input_port']}" or group==r['R_group'])]
            info={m:[r for r in matching if r['method']==m] for m in METHODS}
            for base in ('linear','column_arc','local_tangent'):
                bybase={r['geometry_id']:r for r in info[base]};bytarget={r['geometry_id']:r for r in info['path']}
                for metric in ('rms','weighted_rms','mae'):
                    ids=[k for k in bybase if bybase[k][metric] is not None and bytarget[k][metric] is not None]
                    if not ids:continue
                    a=np.array([bybase[k][metric] for k in ids]);b=np.array([bytarget[k][metric] for k in ids])
                    summaries.append({'frame':frame,'axis':axis,'zone':zone,'band':band,'gate':gate,'group':group,'baseline':base,'metric':metric,'geometries':len(ids),
                        'baseline_mean':float(a.mean()),'path_mean':float(b.mean()),'mean_difference_baseline_minus_path':float((a-b).mean()),
                        'relative_reduction':float(1-b.mean()/a.mean()) if a.mean()>0 else None,
                        'improved':int(np.sum(b<a-1e-10)),'worsened':int(np.sum(b>a+1e-10)),
                        'per_geometry_difference':dict(zip(ids,(a-b).tolist()))})
    save('phase_summary.json',summaries)
    save('diagnostics_complete.json',{'time_utc':now(),'sample_count':16,'wall_seconds':time.monotonic()-t0,'phase_rows':len(rows),
        'carrier_methods':list(METHODS),'formal_data_generated':0,'networks_run':0,'seed':2026,'protocol_sha256':sha(OUT/'PROTOCOL.md'),
        'path_geometry_sha256':sha(OUT/'path_geometry.py'),'diagnostic_sha256':sha(OUT/'diagnose.py')})
    print(json.dumps(calibration,indent=2),flush=True)
if __name__=='__main__':
    for name in ['maps','profiles']:(OUT/name).mkdir(exist_ok=True)
    main()
