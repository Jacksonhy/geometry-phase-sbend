"""One geometry-only C0 extension candidate; no E/H/S or field calibration."""
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
V1=Path(__file__).resolve().parent.parent/'research_sbend_path'
if str(V1) not in sys.path:sys.path.insert(0,str(V1))
import path_geometry as v1

def step5(u):
    u=np.clip(np.asarray(u,dtype=float),0.,1.)
    return u*u*u*(10.+u*(-15.+6.*u))

@lru_cache(maxsize=40)
def quadrature(poly_tuple,dx):
    poly=np.asarray(poly_tuple,dtype=float).reshape(-1,2)
    ext=poly.copy();ext[0,0]-=10*dx;ext[-1,0]+=10*dx
    _,_,arc,_=v1.curve_info(ext)
    count=int(np.ceil(arc[-1]/(dx/4)))
    ss=np.linspace(0.,arc[-1],count+1)
    coords,_=v1.points_on_arc(ss,ext)
    weights=np.ones(len(ss));weights[[0,-1]]=.5
    return coords,ss-10*dx,weights

def soft_arc(points,poly,dx,chunk=384):
    nodes,s,w=quadrature(tuple(np.asarray(poly).ravel()),float(dx))
    flat=np.asarray(points,dtype=float).reshape(-1,2);out=[]
    for start in range(0,len(flat),chunk):
        r=flat[start:start+chunk]
        d2=np.sum((r[:,None,:]-nodes[None,:,:])**2,axis=-1)
        z=-(d2-np.min(d2,axis=1,keepdims=True))/(2*dx*dx)
        weights=np.exp(z)*w
        out.append(np.sum(weights*s,axis=1)/np.sum(weights,axis=1))
    return np.concatenate(out).reshape(np.asarray(points).shape[:-1]) if len(flat) else np.empty(np.asarray(points).shape[:-1])

def distances_at_points(points,poly,source_center_um,input_port,wg_width_um,dx_um=.05):
    if input_port not in (1,2):raise ValueError('input port must be 1 or 2')
    if dx_um<=0 or wg_width_um<=0:raise ValueError('positive physical scales required')
    points=np.asarray(points,dtype=float);poly=np.asarray(poly,dtype=float)
    src=np.asarray(source_center_um,dtype=float)[:2]
    source=v1.project_points(src[None,:],poly)
    if float(source['distance'][0])>.05:raise ValueError('source is off centerline')
    src_s=float(source['s'][0]);direction=1. if input_port==1 else -1.
    _,_,arc,_=v1.curve_info(poly)
    x=points[...,0];z=points[...,1]-np.interp(x,poly[:,0],poly[:,1])
    sc=np.interp(x,poly[:,0],arc);ty=v1.smooth_tangent_at_x(x,poly)[...,1]
    r0=wg_width_um/2+dx_um;r1=r0+.5
    B=1-step5((abs(z)-r0)/(r1-r0))
    A=step5((x-poly[1,0])/dx_um)*step5((poly[-2,0]-x)/dx_um)
    # Q is only needed on the compact nonzero support; at its boundary B=0
    # with zero first/second derivatives. No nearest-point coordinate enters Q.
    active=(A>0)&(B>0);q=sc.copy()
    if np.any(active):q[active]=soft_arc(points[active],poly,dx_um)
    absolute=sc+A*(B*(q-sc)+(1-B)*ty*z)
    return {'path_v2':direction*(absolute-src_s),'blend_core':B,'bend_gate':A,
            'column_offset_um':z,'source_s':src_s,'direction':direction,'r0_um':r0,'r1_um':r1}

def geometry_features(params,x_um,y_um,source_center_um,input_port):
    x=np.asarray(x_um,dtype=float);y=np.asarray(y_um,dtype=float)
    dx=float(np.median(np.diff(x)));dy=float(np.median(np.diff(y)))
    if abs(dx-dy)>1e-10:raise ValueError('this frozen candidate uses square grids')
    poly,bend=v1.centerline_from_params(params,(x[0]-dx/2,x[-1]+dx/2))
    pts=np.stack(np.meshgrid(x,y),axis=-1)
    result=distances_at_points(pts,poly,source_center_um,input_port,params['wg_width_um'],dx)
    result.update({'polyline_um':poly,'bend_um':bend,'dx_um':dx,'dy_um':dy})
    return result
