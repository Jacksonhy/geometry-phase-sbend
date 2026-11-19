"""All fixed diagnostics for the single candidate. No optimization or field fitting."""
import time,sys
import numpy as np
from support import ROOT,OUT,V1,OLD,read,save,csvsave,sha,now
import path_v2 as v2
v1=v2.v1
from diagnose import statistics
METHODS=('linear','local_tangent','path','path_v2')

def evaluate(points,poly,src,ip,w,method):
    if method=='path_v2':return v2.distances_at_points(points,poly,src,ip,w)['path_v2']
    return v1.distances_at_points(points,poly,src,ip)[method]

def invariants(poly,src,ip,w,beta,source_size):
    _,_,arc,t=v1.curve_info(poly)
    p,tang=v1.points_on_arc(np.linspace(.5,arc[-1]-.5,71),poly)
    norm=np.stack([-tang[:,1],tang[:,0]],axis=-1)
    q=np.concatenate([p,p+norm*.2*w,p-norm*.2*w,
        np.stack(np.meshgrid(np.linspace(-11,11,17),np.linspace(-3.9,3.9,7)),axis=-1).reshape(-1,2)])
    f=v2.distances_at_points(q,poly,src,ip,w)['path_v2']
    reverse=v2.distances_at_points(q,poly,src,3-ip,w)['path_v2']
    mirrored=v2.distances_at_points(-q,-poly[::-1],-src,3-ip,w)['path_v2']
    shift=np.array([.7,-.2]);ss=src.copy();ss[:2]+=shift
    translated=v2.distances_at_points(q+shift,poly+shift,ss,ip,w)['path_v2']
    reflected=poly.copy();reflected[:,1]*=-1;rq=q.copy();rq[:,1]*=-1;rs=src.copy();rs[1]*=-1
    mirror_y=v2.distances_at_points(rq,reflected,rs,ip,w)['path_v2']
    sp=np.stack([np.full(101,src[0]),np.linspace(src[1]-source_size/2,src[1]+source_size/2,101)],axis=-1)
    source_error=float(abs(v2.distances_at_points(sp,poly,src,ip,w)['path_v2']).max())
    source_center=float(abs(v2.distances_at_points(src[None,:2],poly,src,ip,w)['path_v2'][0]))
    result={'direction_antisymmetry_um':float(abs(f+reverse).max()),
        'rotation_equivariance_um':float(abs(f-mirrored).max()),'translation_equivariance_um':float(abs(f-translated).max()),
        'y_reflection_equivariance_um':float(abs(f-mirror_y).max()),'source_plane_zero_um':source_error,
        'source_center_zero_um':source_center,'unit_modulus_error':float(abs(abs(np.exp(1j*beta*f))-1).max())}
    result['passed']=all(v<1e-9 for v in result.values())
    return result

def max_edges(gid,ip,poly,src,w,beta,x,y,distances):
    rows=[]
    X,Y=np.meshgrid(x,y);z=abs(Y-np.interp(X,poly[:,0],poly[:,1]))
    r0=w/2+.05;r1=r0+.5
    bands={'all':np.ones_like(z,dtype=bool),'core':z<=w/2,'transition':(z>=r0)&(z<=r1),'far':z>=r1}
    for name,D in distances.items():
        for axis,ax in [('x',1),('y',0)]:
            a=[slice(None)]*2;b=a.copy();a[ax]=slice(None,-1);b[ax]=slice(1,None);a=tuple(a);b=tuple(b)
            diff=D[b]-D[a]
            for band,mask in bands.items():
                valid=mask[a]&mask[b]
                values=np.where(valid,abs(diff),-1)
                j=np.unravel_index(values.argmax(),values.shape);k=list(j);k[ax]+=1;k=tuple(k)
                inc=float(diff[j]);carrier_diff=abs(np.exp(1j*beta*inc)-1)
                rows.append({'geometry_id':gid,'input_port':ip,'method':name,'axis':axis,'band':band,
                    'max_abs_coordinate_gradient':abs(inc)/.05,'max_abs_unwrapped_phase_gradient_rad_um':abs(beta*inc)/.05,
                    'coordinate_increment_um':inc,'unwrapped_phase_increment_rad':beta*inc,
                    'complex_carrier_edge_difference':float(carrier_diff),'x_um':float((x[j[1]]+x[k[1]])/2),
                    'y_um':float((y[j[0]]+y[k[0]])/2),'count_gradient_gt_5':int((valid&(abs(diff)/.05>5)).sum()),
                    'count_gradient_gt_10':int((valid&(abs(diff)/.05>10)).sum())})
    return rows

def probes(gid,poly,src,ip,w,beta):
    old=[r for r in read(V1/'far_field_extension.json') if r['geometry_id']==gid and r['method']=='path']
    worst=max(old,key=lambda r:r['max_abs_coordinate_gradient'])
    ax=0 if worst['axis']=='x' else 1
    mid=np.array([worst['x_um'],worst['y_um']]);delta=np.zeros(2);delta[ax]=.025
    lo=mid-delta;hi=mid+delta
    # Locate the old branch cut geometrically; never select based on new phase quality.
    for _ in range(40):
        mid=(lo+hi)/2
        vals=v1.distances_at_points(np.stack([lo,mid,hi]),poly,src,ip)['path']
        if abs(vals[1]-vals[0])>=abs(vals[2]-vals[1]):hi=mid
        else:lo=mid
    cut=(lo+hi)/2
    x=np.unique(np.r_[np.linspace(-11,11,65),poly[1:-1:8,0]])
    yc=np.interp(x,poly[:,0],poly[:,1])
    r0=w/2+.05;r1=r0+.5
    centers=[];directions=[];kinds=[]
    centers.append(cut);e=np.zeros(2);e[ax]=1;directions.append(e);kinds.append('old_worst_cut')
    for radius,label in [(r0,'r0'),(r1,'r1')]:
        for sign in (-1,1):
            centers.extend(np.column_stack([x,yc+sign*radius]))
            directions.extend(np.tile([0.,1.],(len(x),1)));kinds.extend([label]*len(x))
    for xb in [poly[1,0],poly[1,0]+.05,poly[-2,0]-.05,poly[-2,0]]:
        yc0=np.interp(xb,poly[:,0],poly[:,1])
        for yy in yc0+np.array([-r1,-r0,0,r0,r1]):
            centers.append([xb,yy]);directions.append([1.,0.]);kinds.append('bend_gate')
    for yy in src[1]+np.linspace(-.7,.7,7):
        centers.append([src[0],yy]);directions.append([1.,0.]);kinds.append('source_plane')
    centers=np.asarray(centers);directions=np.asarray(directions);kinds=np.asarray(kinds);rows=[]
    for eps in 10.**np.arange(-2,-9,-1):
        points=np.stack([centers-eps*directions,centers+eps*directions],axis=1)
        base=v1.distances_at_points(points,poly,src,ip)
        val2=v2.distances_at_points(points,poly,src,ip,w)['path_v2']
        for name in METHODS:
            values=val2 if name=='path_v2' else base[name];jump=values[:,1]-values[:,0]
            for kind in np.unique(kinds):
                valid=kinds==kind;i=np.flatnonzero(valid)[np.argmax(abs(jump[valid]))]
                rows.append({'geometry_id':gid,'method':name,'kind':str(kind),'epsilon_um':float(eps),
                    'max_abs_distance_difference_um':float(abs(jump[i])),
                    'apparent_max_gradient':float(abs(jump[i])/(2*eps)),
                    'max_abs_unwrapped_phase_difference_rad':float(abs(beta*jump[i])),
                    'max_complex_carrier_difference':float(np.max(abs(np.exp(1j*beta*jump[valid])-1))),
                    'x_um':float(centers[i,0]),'y_um':float(centers[i,1])})
    return rows

def curvilinear_metrics(gid,ip,w,profile,phases,bend):
    P=profile['points_um'];E=profile['Ez_samples'];ns=profile['n_um'];gate=profile['amplitude_gate']
    X=P[...,0];N=np.broadcast_to(ns[None,:],X.shape)
    zones={'all':(X>=-11)&(X<=11),'bend':(X>=bend[0,0])&(X<=bend[-1,0]),
        'leads':((X<bend[0,0])|(X>bend[-1,0]))&(X>=-11)&(X<=11)}
    bands={'core':abs(N)<=w/2+1e-10,'center':abs(N)<=w/4+1e-10,
        'negative_side':(N>=-w/2-1e-10)&(N<-w/4),
        'positive_side':(N<=w/2+1e-10)&(N>w/4),'halo':(abs(N)>w/2+1e-10)&(abs(N)<=w/2+.5+1e-10)}
    rows=[]
    for axis,ax,step in [('propagation',0,.05),('normal',1,.025)]:
        a=[slice(None)]*2;b=a.copy();a[ax]=slice(None,-1);b[ax]=slice(1,None);a=tuple(a);b=tuple(b)
        weights=abs(E[a])*abs(E[b])
        for method,phi in phases.items():
            g=np.angle(E[b]*E[a].conj()*np.exp(-1j*(phi[b]-phi[a])))/step
            for zone,zm in zones.items():
                for band,bm in bands.items():
                    mask=zm[a]&zm[b]&bm[a]&bm[b]
                    for gating in ('amplitude_1pct','ungated'):
                        use=mask&(gate[a]&gate[b] if gating=='amplitude_1pct' else True)
                        rows.append({'geometry_id':gid,'input_port':ip,'method':method,'frame':'curvilinear','axis':axis,
                            'zone':zone,'band':band,'gate':gating,**statistics(g,weights,use)})
    return rows

def cartesian_metrics(gid,ip,field,epsmap,x,phases,amax):
    rows=[];X=np.broadcast_to(x[None,:],field.shape);core=epsmap>(epsmap.min()+epsmap.max())/2
    gate=abs(field)>=.01*amax;central=(X>=-11)&(X<=11)
    for axis,ax in [('x',1),('y',0)]:
        a=[slice(None)]*2;b=a.copy();a[ax]=slice(None,-1);b[ax]=slice(1,None);a=tuple(a);b=tuple(b)
        weights=abs(field[a])*abs(field[b])
        for method,phi in phases.items():
            g=np.angle(field[b]*field[a].conj()*np.exp(-1j*(phi[b]-phi[a])))/.05
            for band,mask in [('core',core),('whole',np.ones_like(core))]:
                valid=mask[a]&mask[b]&central[a]&central[b]
                for gating in ('amplitude_1pct','ungated'):
                    rows.append({'geometry_id':gid,'input_port':ip,'method':method,'frame':'cartesian','axis':axis,
                        'zone':'all','band':band,'gate':gating,**statistics(g,weights,valid&(gate[a]&gate[b] if gating=='amplitude_1pct' else True))})
    return rows

def main():
    started=time.monotonic();records=read(OLD/'sbend_index.json')
    freeze=read(OUT/'candidate_freeze.json')
    assert all(sha(OUT/n)==h for n,h in freeze['sources'].items())
    # Analytic straight limit, with the same lead+bend representation.
    straight=np.array([[-12,.3],[-6,.3],[6,.3],[12,.3]])
    q=np.stack(np.meshgrid(np.linspace(-11,11,33),np.linspace(-3,3,9)),axis=-1)
    analytic=max(float(abs(v2.distances_at_points(q,straight,[src,.3,0],ip,.5)['path_v2']-
        (q[...,0]-src)*(1 if ip==1 else -1)).max()) for src,ip in [(-11.9,1),(11.9,2)])
    assert analytic<1e-9
    rows=[];edges=[];inv=[];continuity=[];deviations=[];runtimes=[]
    (OUT/'maps').mkdir(exist_ok=True);(OUT/'profiles').mkdir(exist_ok=True)
    oldmetrics=read(V1/'phase_summary.json')
    for rec in records:
        t0=time.monotonic();gid=rec['geometry_id'];w=rec['params']['wg_width_um'];ip=rec['input_port']
        with np.load(V1/'maps'/f'{gid}.npz') as m,np.load(V1/'profiles'/f'{gid}.npz') as p,np.load(OLD/rec['sample']) as raw:
            x=m['x_um'];y=m['y_um'];poly=m['polyline_um'];bend=m['bend_um'];src=m['source_center_um'];beta=float(m['beta_Y_rad_um'])
            # Geometry alone constructs the candidate before any residual metrics.
            mapstart=time.monotonic();features=v2.geometry_features(rec['params'],x,y,src,ip);D2=features['path_v2'];mapsec=time.monotonic()-mapstart
            distances={name:m[name+'_um'] for name in METHODS if name!='path_v2'};distances['path_v2']=D2
            edges.extend(max_edges(gid,ip,poly,src,w,beta,x,y,distances))
            inv.append({'geometry_id':gid,**invariants(poly,src,ip,w,beta,float(raw['source_size_um'][1]))})
            continuity.extend(probes(gid,poly,src,ip,w,beta))
            P=p['points_um'];fp=v2.distances_at_points(P,poly,src,ip,w);phi2=beta*fp['path_v2']
            phases={name:p[name+'_phase'] for name in METHODS if name!='path_v2'};phases['path_v2']=phi2
            rows.extend(curvilinear_metrics(gid,ip,w,p,phases,bend))
            E=p['Ez_samples'];amax=float(abs(E[:,np.argmin(abs(p['n_um']))]).max())
            rows.extend(cartesian_metrics(gid,ip,raw['Ez'],raw['eps'],x,{k:beta*v for k,v in distances.items()},amax))
            nmask=(abs(p['n_um'][None,:])<=w/2+1e-10)&(P[...,0]>=-11)&(P[...,0]<=11)
            dp=phi2-p['path_phase']
            deviations.append({'geometry_id':gid,'input_port':ip,
                'max_abs_core_phase_difference_vs_v1_rad':float(abs(dp[nmask]).max()),
                'rms_core_phase_difference_vs_v1_rad':float(np.sqrt(np.mean(dp[nmask]**2))),
                'max_abs_centerline_phase_difference_vs_v1_rad':float(abs(dp[:,np.argmin(abs(p['n_um']))]).max())})
            np.savez_compressed(OUT/'maps'/f'{gid}.npz',x_um=x,y_um=y,path_v2_um=D2,path_v2_carrier=np.exp(1j*beta*D2).astype(np.complex64),
                blend_core=features['blend_core'],bend_gate=features['bend_gate'],source_center_um=src,beta_Y_rad_um=beta)
            np.savez_compressed(OUT/'profiles'/f'{gid}.npz',path_v2_phase=phi2,path_v2_um=fp['path_v2'])
        runtimes.append({'geometry_id':gid,'candidate_grid_seconds':mapsec,'all_diagnostics_seconds':time.monotonic()-t0})
        print(f'{len(runtimes)}/16 {gid} map={mapsec:.2f}s all={runtimes[-1]["all_diagnostics_seconds"]:.2f}s',flush=True)
    # Sanity: same cached fields and definition must reproduce the old diagnostics.
    oldcsv=[]
    import csv
    with (V1/'phase_diagnostics.csv').open(encoding='utf-8-sig') as f:oldcsv=list(csv.DictReader(f))
    keys=('geometry_id','method','frame','axis','zone','band','gate')
    lookup={tuple(r[k] for k in keys):r for r in oldcsv}
    reproduce=max(abs(r['weighted_rms']-float(lookup[tuple(r[k] for k in keys)]['weighted_rms']))
        for r in rows if r['method']!='path_v2' and r['weighted_rms'] is not None)
    assert reproduce<1e-10,reproduce
    summary=[]
    for frame,axis,zone,band,gate in sorted({tuple(r[k] for k in ('frame','axis','zone','band','gate')) for r in rows}):
        for group in ('all','input1','input2'):
            subset=[r for r in rows if tuple(r[k] for k in ('frame','axis','zone','band','gate'))==(frame,axis,zone,band,gate)
                and (group=='all' or group==f'input{r["input_port"]}')]
            target={r['geometry_id']:r for r in subset if r['method']=='path_v2'}
            for base in ('path','local_tangent','linear'):
                bb={r['geometry_id']:r for r in subset if r['method']==base}
                ids=[i for i in target if target[i]['weighted_rms'] is not None and bb[i]['weighted_rms'] is not None]
                if not ids:continue
                a=np.array([bb[i]['weighted_rms'] for i in ids]);b=np.array([target[i]['weighted_rms'] for i in ids])
                summary.append({'frame':frame,'axis':axis,'zone':zone,'band':band,'gate':gate,'group':group,'baseline':base,
                    'geometries':len(ids),'baseline_mean':float(a.mean()),'v2_mean':float(b.mean()),
                    'relative_reduction':float(1-b.mean()/a.mean()),'improved':int((b<a-1e-10).sum()),'worsened':int((b>a+1e-10).sum()),
                    'per_geometry_relative_change':{i:float(bj/aj-1) for i,aj,bj in zip(ids,a,b)}})
    geomok=all(r['passed'] for r in inv)
    gradmax=max(r['max_abs_coordinate_gradient'] for r in edges if r['method']=='path_v2' and r['band']=='all')
    core=[r for r in summary if r['frame']=='curvilinear' and r['zone']=='all' and r['band']=='core'
        and r['gate']=='amplitude_1pct' and r['group']=='all' and r['baseline']=='path']
    maxdeviation=max(r['max_abs_core_phase_difference_vs_v1_rad'] for r in deviations)
    maxdegrade=max(v for r in core for v in r['per_geometry_relative_change'].values())
    continuity_small=[r for r in continuity if r['method']=='path_v2' and r['epsilon_um']<1.01e-8]
    max_small_gradient=max(r['apparent_max_gradient'] for r in continuity_small)
    decision={'time_utc':now(),'candidate_count':1,'seed':2026,'geometry_invariants_passed':geomok,
        'analytic_straight_max_error_um':analytic,'old_metrics_max_reproduction_error':reproduce,
        'grid_max_coordinate_gradient':gradmax,'small_epsilon_probe_max_gradient':max_small_gradient,
        'max_core_phase_difference_vs_v1_rad':maxdeviation,'worst_per_geometry_core_rms_relative_degradation':maxdegrade,
        'passes_declared_checks':bool(geomok and gradmax<=5 and max_small_gradient<=5 and maxdeviation<=.1 and maxdegrade<=.02),
        'scope':'16 seen development geometries; no new FDTD/training/inference/test','wall_seconds':time.monotonic()-started,
        'continuity_claim':'C0 by positive finite Gaussian quadrature and continuous gates; not globally C1/C2'}
    for name,contents in [('phase_diagnostics',rows),('phase_summary',summary),('gradient_checks',edges),
        ('geometry_invariants',inv),('continuity_probes',continuity),('core_deviation',deviations),('runtime',runtimes)]:
        save(name+'.json',contents);csvsave(name+'.csv',contents)
    save('decision.json',decision)
    print(decision,flush=True)
if __name__=='__main__':main()
