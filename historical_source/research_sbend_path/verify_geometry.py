"""Independent meaningful coordinate invariants on synthetic and actual geometry."""
import inspect,json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from common import OUT,OLD,read,save
from path_geometry import geometry_features,centerline_from_params,curve_info,project_points,points_on_arc,distances_at_points,carriers_from_features

def main():
    # Analytic straight and tilted lines verify arc and 2D projection, not code mirroring.
    straight=np.array([[-12.,.3],[12.,.3]])
    pts=np.array([[2.,.9],[-3.,-.4],[0.,.3]])
    L=distances_at_points(pts,straight,[-11.9,.3,0],1)
    assert np.allclose(L['path'],pts[:,0]+11.9,atol=1e-12)
    R=distances_at_points(pts,straight,[11.9,.3,0],2)
    assert np.allclose(R['path'],11.9-pts[:,0],atol=1e-12)
    a=np.array([-2.,-1.]);b=np.array([2.,1.]);tilted=np.vstack([a,b]);t=(b-a)/np.linalg.norm(b-a);n=np.array([-t[1],t[0]])
    q=a+np.array([.5,1.,2.])[:,None]*t+np.array([-.3,.2,.7])[:,None]*n
    pr=project_points(q,tilted)
    assert np.allclose(pr['s'],[.5,1.,2.],atol=1e-12)
    assert np.allclose(pr['normal_offset'],[-.3,.2,.7],atol=1e-12)
    rows=[]
    for rec in read(OLD/'sbend_index.json'):
        gid=rec['geometry_id'];p=rec['params'];ip=rec['input_port']
        with np.load(OLD/rec['sample'],allow_pickle=False) as d:
            x=d['x_um'];y=d['y_um'];src=d['source_center_um'];eps=d['eps'];poly,bend=centerline_from_params(p)
            _,length,arc,t=curve_info(poly)
            pr=project_points(poly,poly);arc_err=float(abs(pr['s']-arc).max())
            assert arc_err<1e-11 and np.all(np.diff(arc)>0) and arc[-1]>=np.linalg.norm(poly[-1]-poly[0])
            f=distances_at_points(np.asarray(src)[None,:2],poly,src,ip)
            assert max(abs(float(f[k][0])) for k in ('path','linear','column_arc','local_tangent'))<1e-11
            q=poly[1:-1][::12];same=distances_at_points(q,poly,src,ip);reverse=distances_at_points(q,poly,src,3-ip)
            assert np.max(abs(same['path']+reverse['path']))<1e-11
            # Joint (x,y)->(-x,-y), reverse polyline order, swap physical input side.
            mirrored=-poly[::-1];mq=-q;msrc=-src
            mf=distances_at_points(mq,mirrored,msrc,3-ip)
            mirror_err=float(abs(same['path']-mf['path']).max());assert mirror_err<1e-10
            # Mid-segment normals within the core have correct arc projection.
            mids=.5*(poly[:-1]+poly[1:]);norm=np.stack([-t[:,1],t[:,0]],axis=1)
            qq=mids+norm*.2*p['wg_width_um'];jp=project_points(qq,poly)
            nerr=float(abs(jp['s']-(arc[:-1]+length/2)).max());assert nerr<1e-8
            # Geometry-vs-raster checks independent of Ez; centerline stays inside material core.
            sample_s=np.linspace(0,arc[-1],3000);q,_=points_on_arc(sample_s,poly)
            valid=(q[:,0]>=x[0])&(q[:,0]<=x[-1]);q=q[valid]
            interp=RegularGridInterpolator((y,x),eps,bounds_error=False,fill_value=np.nan)
            on=interp(q[:,[1,0]]);corethreshold=(eps.min()+eps.max())/2
            coverage=float(np.mean(on>corethreshold));assert coverage>.99
            # Raster center estimate gives limited resolution check, not a second path implementation.
            threshold=eps>(eps.min()+eps.max())/2
            yc=np.array([np.mean(y[threshold[:,j]]) for j in range(len(x))]);exact=np.interp(x,poly[:,0],poly[:,1])
            ce=abs(yc-exact)
            # Saved maps have the correct same reference / unit carrier for all methods.
            with np.load(OUT/'maps'/f'{gid}.npz') as maps:
                moderr=max(float(np.max(abs(abs(maps[k+'_carrier'])-1))) for k in ('linear','column_arc','local_tangent','path'))
                assert moderr<2e-7
                grid=np.stack(np.meshgrid(x[::23],y[::19]),axis=-1)
                check=distances_at_points(grid,poly,src,ip)
                assert np.max(abs(check['path']-maps['path_um'][::19,::23]))<1e-11
            rows.append({'geometry_id':gid,'input_port':ip,'centerline_arc_consistency_um':arc_err,'source_reference_zero':True,
                'same_source_direction_antisymmetry':True,'mirror_equivariance_max_um':mirror_err,'core_normal_projection_max_s_error_um':nerr,
                'centerline_eps_core_coverage':coverage,'raster_centerline_rms_um':float(np.sqrt(np.mean(ce*ce))),'raster_centerline_max_um':float(ce.max()),
                'unit_carrier_error_float32':moderr,'geometry_inputs_only':True})
    signatures={k:str(inspect.signature(f)) for k,f in [('geometry_features',geometry_features),('distances_at_points',distances_at_points),('carriers_from_features',carriers_from_features)]}
    save('geometry_verification.json',{'passed':True,'synthetic_straight_and_tilted':True,'actual_geometries':len(rows),'signatures':signatures,
        'notes':'invariants apply in single centerline core tube; no global eikonal claim in cladding','records':rows})
    print(json.dumps({'passed':True,'geometries':len(rows),'max_raster_centerline_error_um':max(r['raster_centerline_max_um'] for r in rows)},indent=2))
if __name__=='__main__':main()
