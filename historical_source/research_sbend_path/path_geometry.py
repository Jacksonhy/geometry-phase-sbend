"""Geometry-only S-bend arc coordinates. No E/H/S or query-dependent calibration."""
import ast
from functools import lru_cache
import numpy as np
from common import SRC

@lru_cache(None)
def official_euler_function():
    path=SRC/'FDTD/sbend/sbend.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='euler_sbend_points')
    env={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env['euler_sbend_points']

def centerline_from_params(params,x_bounds=(-12.,12.)):
    """Same 256-point polyline used by official geometry, with straight leads."""
    xb,yb=official_euler_function()(params['lateral_offset_um'],params['R_min_um'],n_pts=256)
    xb=xb-.5*(xb.min()+xb.max());yb=yb-.5*(yb.min()+yb.max())
    bend=np.column_stack([xb,yb])
    if not (x_bounds[0]<xb[0]<xb[-1]<x_bounds[1]):raise ValueError('bend does not fit supplied domain')
    poly=np.vstack([[x_bounds[0],yb[0]],bend,[x_bounds[1],yb[-1]]])
    return poly,bend

def curve_info(poly):
    poly=np.asarray(poly,dtype=np.float64)
    seg=np.diff(poly,axis=0);length=np.linalg.norm(seg,axis=1)
    if np.any(length<=0) or np.any(np.diff(poly[:,0])<=0):raise ValueError('requires single nondegenerate x-monotone centerline')
    arc=np.r_[0.,np.cumsum(length)];tangent=seg/length[:,None]
    return seg,length,arc,tangent

def project_points(points,poly,chunk=1024):
    """Exact nearest line-segment projection, deterministic first-segment ties."""
    points=np.asarray(points,dtype=np.float64);shape=points.shape[:-1];flat=points.reshape(-1,2)
    seg,length,arc,tangent=curve_info(poly);result=[]
    for start in range(0,len(flat),chunk):
        q=flat[start:start+chunk];delta=q[:,None,:]-poly[:-1][None,:,:]
        t=np.clip(np.einsum('nji,ji->nj',delta,seg)/length[None,:]**2,0,1)
        projected=poly[None,:-1,:]+t[:,:,None]*seg[None,:,:]
        dist2=np.sum((q[:,None,:]-projected)**2,axis=2)
        index=np.argmin(dist2,axis=1);j=np.arange(len(q));p=projected[j,index];tau=t[j,index]
        s=arc[index]+tau*length[index]
        normal=np.column_stack([-tangent[index,1],tangent[index,0]])
        signed=np.sum((q-p)*normal,axis=1)
        result.append(np.column_stack([s,p,signed,index,np.sqrt(dist2[j,index])]))
    a=np.concatenate(result).reshape(*shape,6)
    return {'s':a[...,0],'closest':a[...,1:3],'normal_offset':a[...,3],
            'segment_index':a[...,4].astype(int),'distance':a[...,5]}

def points_on_arc(s,poly):
    _,length,arc,tangent=curve_info(poly);s=np.asarray(s,dtype=np.float64)
    idx=np.clip(np.searchsorted(arc,s,side='right')-1,0,len(length)-1)
    frac=(s-arc[idx])/length[idx]
    p=poly[idx]+frac[...,None]*np.diff(poly,axis=0)[idx]
    return p,tangent[idx]

def smooth_tangent_at_x(x,poly):
    # Central derivative at vertices; no truth-field smoothing or tuned parameter.
    slope=np.gradient(poly[:,1],poly[:,0],edge_order=1)
    v=np.interp(x,poly[:,0],slope)
    return np.stack([np.ones_like(v),v],axis=-1)/np.sqrt(1+v*v)[...,None]

def distances_at_points(points,poly,source_center_um,input_port):
    """Physical units μm. Source projection independent of clipped src_mask."""
    points=np.asarray(points,dtype=np.float64)
    if input_port not in (1,2):raise ValueError('S-bend has input ports 1 and 2')
    direction=1. if input_port==1 else -1.
    source=np.asarray(source_center_um,dtype=float)[:2]
    source_projection=project_points(source[None,:],poly)
    if float(source_projection['distance'][0])>.05:raise ValueError('source center is not on centerline within one cell')
    src_s=float(source_projection['s'][0])
    proj=project_points(points,poly)
    _,_,arc,_=curve_info(poly)
    xc=points[...,0];yc=np.interp(xc,poly[:,0],poly[:,1]);sc=np.interp(xc,poly[:,0],arc)
    tangent=smooth_tangent_at_x(xc,poly)
    return {'path':direction*(proj['s']-src_s),
            'linear':direction*(xc-source[0]),
            'column_arc':direction*(sc-src_s),
            'local_tangent':direction*(sc+tangent[...,1]*(points[...,1]-yc)-src_s),
            'projection':proj,'source_s':src_s,'direction':direction}

def geometry_features(params,x_um,y_um,source_center_um,input_port):
    """Public field-free adapter; only geometry/grid/source enter."""
    x=np.asarray(x_um,dtype=float);y=np.asarray(y_um,dtype=float)
    dx=float(np.median(np.diff(x)));dy=float(np.median(np.diff(y)))
    poly,bend=centerline_from_params(params,(float(x[0]-dx/2),float(x[-1]+dx/2)))
    xx,yy=np.meshgrid(x,y);points=np.stack([xx,yy],axis=-1)
    result=distances_at_points(points,poly,source_center_um,input_port)
    result.update({'polyline_um':poly,'bend_um':bend,'x_um':x,'y_um':y,'dx_um':dx,'dy_um':dy})
    return result

def beta_from_training_calibration(eps_max,dx_um,coefficients):
    """Historical convention rad/pixel converted to rad/μm; no target access."""
    beta=float(np.polyval(coefficients,float(eps_max))/float(dx_um))
    if beta<=0:raise ValueError('nonpositive propagation constant')
    return beta

def carriers_from_features(features,beta_rad_um):
    if not np.isfinite(beta_rad_um) or beta_rad_um<=0:raise ValueError('invalid beta')
    return {name:np.exp(1j*beta_rad_um*features[name]) for name in ('linear','column_arc','local_tangent','path')}
