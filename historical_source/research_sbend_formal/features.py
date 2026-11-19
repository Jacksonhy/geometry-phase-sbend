"""Field-free common covariates; only output carrier differs between methods."""
import time
import numpy as np
from common import v1,v2,METHODS

def geometry(params,x,y,source,input_port):
    t=time.perf_counter();f=v1.geometry_features(params,x,y,source,input_port);t1=time.perf_counter()
    g=v2.geometry_features(params,x,y,source,input_port);t2=time.perf_counter()
    d={'direct':np.zeros_like(f['linear']),'linear':f['linear'],'local_tangent':f['local_tangent'],
       'path_v1':f['path'],'path_v2':g['path_v2']}
    assert all(np.isfinite(a).all() for a in d.values())
    grad=max(float(abs(np.diff(d['path_v2'],axis=ax)).max()/.05) for ax in [0,1])
    return d,{'v1_and_baselines_seconds':t1-t,'v2_seconds':t2-t1,'v2_max_axis_gradient':grad},f

def pack(a):
    a=np.asarray(a,dtype=np.float32);assert a.shape[-3]==12
    z=np.zeros((*a.shape[:-3],7,*a.shape[-2:]),np.complex64);z[...,0,:,:]=a[...,0,:,:]
    for j in range(1,12):
        c=1+(j-1)//2
        if j%2:z[...,c,:,:].real=a[...,j,:,:]
        else:z[...,c,:,:].imag=a[...,j,:,:]
    return z
def unpack(z):
    a=[z[...,0,:,:].real]
    for j in range(1,12):
        c=1+(j-1)//2;a.append(z[...,c,:,:].real if j%2 else z[...,c,:,:].imag)
    return np.stack(a,axis=-3)
def inputs(eps,src,x,y,source,input_port,beta,d):
    X,Y=np.meshgrid(x,y);shape=eps.shape;full=lambda a:np.full(shape,a)
    a=np.stack([(eps-2.0851359)/4.2,src,X/12,Y/4,full(source[0]/12),full(source[1]/4),
        full(1 if input_port==1 else -1),full(beta*.05),d['path_v1']/24,d['linear']/24,
        d['local_tangent']/24,d['path_v2']/24]).astype(np.float32)
    z=pack(a);assert np.array_equal(a,unpack(z))
    assert np.all(z[0].imag==0) and np.all(z[6].imag==0)
    c={m:np.exp(1j*beta*d[m]).astype(np.complex64)[None] for m in METHODS}
    return z,c

def sample_load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in ['eps','src_mask','Ez','source_center_um','input_port','x_um','y_um']}
