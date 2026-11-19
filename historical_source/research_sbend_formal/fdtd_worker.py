"""Finite serial FDTD audit; original PIC-Flow modules are imported read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLBACKEND','Agg')
import meep as mp
from meep import mpb
import ast,sys,json,time,resource,platform,traceback
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal

ROOT=Path('/mnt/d/game');OUT=ROOT/'research_sbend_formal'
SRC=next(p.parent for p in (ROOT/'PIC-Flow-新研究资料').rglob('README.md') if (p.parent/'FDTD/unified_sweep.py').exists())
sys.path.insert(0,str(SRC/'FDTD'))
from ybranch.ybranch import YBranch2D
from sbend.sbend import EulerSBend2D

def save(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def cjson(a):
    a=np.asarray(a);return {'real':a.real.tolist(),'imag':a.imag.tolist()}
def vec(v):
    return [float(getattr(v,k)() if callable(getattr(v,k)) else getattr(v,k)) for k in ('x','y','z')]
def volume(v):
    if callable(v.center):
        return {'center_um':vec(v.center()),'size_um':[float(v.in_direction(d)) for d in (mp.X,mp.Y,mp.Z)]}
    return {'center_um':vec(v.center),'size_um':vec(v.size)}
def extract_functions(path,names,env):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(body)==len(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),env)
env={'np':np,'eigh_tridiagonal':eigh_tridiagonal}
extract_functions(ROOT/'research_official_audit/validate_sparams.py',['fd_directional_proxy'],env)
proxy=env['fd_directional_proxy']
from typing import Dict,Tuple
maskenv={'np':np,'Dict':Dict,'Tuple':Tuple}
extract_functions(SRC/'FDTD/unified_sweep.py',['_draw_thick_line_mask','_meep_coord_to_cropped_px','_volume_to_line_px','get_device_masks'],maskenv)

def mode_audit(sim,mon,fcen,idx,arrays):
    """Independent numerical E/H overlap on exported interpolated monitor grids.

    P=Re integral(E_m* cross H_m). Unit-power amplitudes follow reciprocal
    overlap. Native alpha is saved as reference, never used to adjust overlaps.
    """
    res=sim.get_eigenmode_coefficients(mon,[1],eig_parity=mp.NO_PARITY)
    alpha=np.asarray(res.alpha[0,0]).copy()
    md=sim.get_eigenmode(fcen,mp.X,mon.where,1,mp.Vector3(),parity=mp.NO_PARITY)
    x,y,z,w=sim.get_array_metadata(dft_cell=mon)
    fe=np.asarray(sim.get_dft_array(mon,mp.Ez,0)).reshape(-1)
    fh=np.asarray(sim.get_dft_array(mon,mp.Hy,0)).reshape(-1)
    assert fe.size==len(y)==fh.size
    em=np.array([md.amplitude(mp.Vector3(float(x[0]),float(yi),0),mp.Ez) for yi in y])
    hm=np.array([md.amplitude(mp.Vector3(float(x[0]),float(yi),0),mp.Hy) for yi in y])
    hx=np.array([md.amplitude(mp.Vector3(float(x[0]),float(yi),0),mp.Hx) for yi in y])
    weights=np.asarray(w).reshape(-1)
    power=float(-np.real(np.sum(weights*em.conj()*hm)))
    if power<=0:raise RuntimeError(f'nonpositive +x mode power at port {idx}: {power}')
    e_h=-np.sum(weights*em.conj()*fh)
    h_e=-np.sum(weights*fe*hm.conj())
    eh=np.array([e_h+h_e,h_e-e_h])/(2*np.sqrt(power))
    # Reverse mode: same Ez, reversed transverse Hy for reciprocal 2D Ez mode.
    for name,value in {'y_um':y,'x_um':x,'weights':weights,'field_Ez':fe,'field_Hy':fh,
                       'mode_Ez':em,'mode_Hx':hx,'mode_Hy':hm,'alpha_native':alpha,'alpha_EH_grid':eh}.items():
        arrays[f'port{idx}/{name}']=value
    # Low-level solver overlap is diagnostic only: not an independent grid reconstruction.
    low={}
    try:
        ov=np.zeros(2,dtype=np.complex128);norm=np.zeros(2,dtype=np.complex128)
        sim.fields.get_mode_flux_overlap(md.swigobj,mon.swigobj,0,ov)
        sim.fields.get_mode_mode_overlap(md.swigobj,md.swigobj,mon.swigobj,norm)
        low={'field_overlap':cjson(ov),'mode_overlap':cjson(norm)}
        arrays[f'port{idx}/lowlevel_field_overlap']=ov;arrays[f'port{idx}/lowlevel_mode_overlap']=norm
    except (TypeError,AttributeError) as exc:low={'not_exposed_for_numpy':str(exc)}
    return alpha,eh,{'port':idx,'monitor':volume(mon.where),'alpha_native':cjson(alpha),'alpha_EH_grid':cjson(eh),
        'vgrp':np.asarray(res.vgrp).tolist(),'cscale':np.asarray(res.cscale).tolist(),
        'kpoints':[vec(k) for k in res.kpoints],'kdom':[vec(k) for k in res.kdom],
        'independent_mode_k':vec(md.k),'independent_mode_vgrp':float(md.group_velocity),
        'mode_power_grid':power,'mode_Ez_max':float(abs(em).max()),'overlap_backend_diagnostic':low,
        'alpha_EH_abs_difference':abs(eh-alpha).tolist(),
        'formula':'P=-Re sum(w*Em.conj()*Hm); alpha+=( -sum(w*Em.conj()*Hf)-sum(w*Ef*Hm.conj()) )/(2sqrt(P)); alpha-=( -sum(w*Ef*Hm.conj())+sum(w*Em.conj()*Hf) )/(2sqrt(P))'}

def build_y(record):
    dev=YBranch2D(**record['params'],wavelength_um=1.55,resolution=20,dpml=1,
        cell_x_um=26,cell_y_um=10,quantize_grid=False,fit_margin_um=.5)
    ip=record['input_port'];src=getattr(dev,f'src_{ip}');fcen=1/1.55
    source=mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=.1*fcen),volume=src,
        eig_band=1,eig_parity=mp.NO_PARITY,eig_match_freq=True,eig_kpoint=mp.Vector3(1 if ip==1 else -1,0,0))
    sim=mp.Simulation(cell_size=dev.cell,resolution=dev.resolution,boundary_layers=[mp.PML(dev.dpml)],
        geometry=dev.geometry,default_material=dev.clad_medium,sources=[source])
    ports=[getattr(dev,f'port_{i}') for i in (1,2,3)]
    monitors=[sim.add_mode_monitor(fcen,0,1,mp.ModeRegion(volume=p,direction=mp.X)) for p in ports]
    dft=sim.add_dft_fields([mp.Ez,mp.Hx,mp.Hy],fcen,0,1,center=dev.full_plane.center,size=dev.full_plane.size)
    return dev,sim,monitors,dft,dft,src

def run_device(family,index):
    t0=time.monotonic()
    listing=json.loads((OUT/'geometry_manifest.json').read_text())
    rec=listing['records'][index];gid=rec['geometry_id'];ip=rec['input_port']
    dest=OUT/'data'/gid;dest.mkdir(parents=True,exist_ok=False)
    save(dest/'started.json',{'record':rec,'unix_time':time.time()})
    if family=='ybranch':dev,sim,mons,dft,dft_h,src=build_y(rec)
    else:
        p=rec['params']
        dev=EulerSBend2D(wg_width=p['wg_width_um'],lateral_offset_um=p['lateral_offset_um'],R_min_um=p['R_min_um'],
            wavelength_um=1.55,resolution=20,dpml=1,cell_x=26,cell_y=10)
        assert dev.n_core>dev.n_clad
        sim,mons,dft,_=dev._build_sim_single(input_port=ip)
        src=mp.Volume(center=sim.sources[0].center,size=sim.sources[0].size)
        dft_h=sim.add_dft_fields([mp.Hx,mp.Hy],1/1.55,0,1,center=dev.full_plane.center,size=dev.full_plane.size)
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6))
    arrays={};alpha=[];eh=[];modes=[]
    for i,mon in enumerate(mons,1):
        a,b,m=mode_audit(sim,mon,1/1.55,i,arrays);alpha.append(a);eh.append(b);modes.append(m)
    alpha=np.array(alpha);eh=np.array(eh)
    ids=np.arange(1,len(mons)+1,dtype=np.int32);outdirs=np.array([1]+[0]*(len(ids)-1))
    ind=ip-1;indir=0 if ip==1 else 1
    S=alpha[np.arange(len(ids)),outdirs]/alpha[ind,indir]
    S_eh=eh[np.arange(len(ids)),outdirs]/eh[ind,indir]
    eps=sim.get_epsilon().T
    E=sim.get_dft_array(dft,mp.Ez,0).T
    Hx=sim.get_dft_array(dft_h,mp.Hx,0).T;Hy=sim.get_dft_array(dft_h,mp.Hy,0).T
    fullmeta=sim.get_array_metadata(dft_cell=dft)
    arrays.update({'full/x_um':fullmeta.x,'full/y_um':fullmeta.y,'full/weights':fullmeta.w.T})
    arrays.update({'full/eps':eps,'full/Ez':E,'full/Hx':Hx,'full/Hy':Hy})
    eps=eps[20:-20,20:-20];E=E[20:-20,20:-20];Hx=Hx[20:-20,20:-20];Hy=Hy[20:-20,20:-20]
    assert E.shape==eps.shape==Hx.shape==Hy.shape==(160,480),(E.shape,eps.shape,Hx.shape,Hy.shape)
    sm,maskids,pm=maskenv['get_device_masks'](dev,family,ip,26,10,1,20,160,480,3)
    assert np.array_equal(ids,maskids)
    if family=='sbend':
        arrays['src_mask_official_old_export']=sm.copy()
        # Use actual DFT voxel-center coordinates, including the half-pixel offset.
        # Historical rasterizer is retained above for auditing, not reused as truth.
        xx,yy=np.meshgrid(fullmeta.x[20:-20],fullmeta.y[20:-20])
        cx,cy=float(src.center.x),float(src.center.y);half=float(src.size.y)/2
        distance2=(xx-cx)**2+(yy-np.clip(yy,cy-half,cy+half))**2
        sm=(distance2<=(3/20)**2+1e-14).astype(np.float32)
        # An exact-plane interpolation mask accompanies the clipped thick label.
        xs=fullmeta.x[20:-20];j=int(np.searchsorted(xs,cx));left=max(0,j-1);right=min(len(xs)-1,j)
        line=np.zeros_like(sm);span=(fullmeta.y[20:-20]>=cy-half)&(fullmeta.y[20:-20]<=cy+half)
        frac=(cx-xs[left])/(xs[right]-xs[left]) if right!=left else 0
        line[span,left]=1-frac;line[span,right]+=frac
        arrays['src_plane_interpolation_weights']=line
    S_ez=proxy(E,eps,pm,ids,ip,1.55,.05)
    qc={'shape':list(E.shape),'finite':all(bool(np.isfinite(a).all()) for a in [E,Hx,Hy,eps,S,S_ez,S_eh]),
        'Ez_max':float(abs(E).max()),'Hx_max':float(abs(Hx).max()),'Hy_max':float(abs(Hy).max()),
        'eps_min':float(eps.min()),'eps_max':float(eps.max()),'guided_power_sum':float(np.sum(abs(S)**2)),
        'source_mask_count':int(sm.sum()),'port_mask_counts':pm.sum(axis=(1,2)).astype(int).tolist(),
        'source_core_overlap':int(np.sum((sm>0)&(eps>(eps.min()+eps.max())/2))),
        'port_core_overlaps':np.sum((pm>0)&(eps[None]>(eps.min()+eps.max())/2),axis=(1,2)).astype(int).tolist(),
        'incoming_alpha_abs':float(abs(alpha[ind,indir]))}
    qc['passed']=bool(qc['finite'] and qc['Ez_max']>1e-6 and qc['guided_power_sum']<=1.5 and qc['source_core_overlap']>0 and min(qc['port_core_overlaps'])>0 and qc['incoming_alpha_abs']>1e-9)
    arrays.update({'geometry_id':np.array(gid),'device':np.array(family),'split':np.array(rec['split']),
        'input_port':np.int32(ip),'wavelength_um':np.float64(1.55),'nx':np.int32(480),'ny':np.int32(160),
        'Lx_um':np.float64(24),'Ly_um':np.float64(8),'dx_um':np.float64(.05),'dy_um':np.float64(.05),
        'resolution':np.int32(20),'eps':eps.astype(np.float32),'Ez':E,'Hx':Hx,'Hy':Hy,
        'Ez_real':E.real.astype(np.float32),'Ez_imag':E.imag.astype(np.float32),'src_mask':sm,'port_ids':ids,'port_masks':pm,
        'alpha_native':alpha,'alpha_EH_grid':eh,'S_native':S,'S_EH_grid':S_eh,'S_Ez_proxy':S_ez,
        'source_center_um':np.array(vec(src.center)),'source_size_um':np.array(vec(src.size)),
        'meep_version':np.array(mp.__version__),'picflow_source_commit':np.array('fb752f74fdb3c1cee4db8e6fd6a0bb6cac616f34'),
        'x_um':fullmeta.x[20:-20],'y_um':fullmeta.y[20:-20]})
    for p,s in zip(ids,S):arrays.update({f'sparams/S{p}{ip}_real':np.float64(s.real),f'sparams/S{p}{ip}_imag':np.float64(s.imag)})
    arrays.update({f'params/{k}':np.float64(v) for k,v in rec['params'].items()})
    np.savez_compressed(dest/'sample.npz',**arrays)
    result={'family':family,'geometry_id':gid,'candidate_index':index,'input_port':ip,'record':rec,
        'source':volume(src),'monitors':modes,'qc':qc,'S_native':cjson(S),'S_EH_grid':cjson(S_eh),'S_Ez_proxy':cjson(S_ez),
        'meep_version':mp.__version__,'normalization':'native |alpha|^2=modal power; S=outgoing alpha/incoming alpha; no deembedding, no phase fitting',
        'proxy_unchanged':True,'EH_method':'quadrature of exported interpolated monitor E/H with separately solved MPB mode, reciprocal forward/backward projection',
        'decay_tol':1e-6,'simulation_time':float(sim.meep_time()),'n_core':float(dev.n_core),'n_clad':float(dev.n_clad),
        'wall_seconds':time.monotonic()-t0,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'native_fields_complex_dtype':str(E.dtype),'source_mask_convention':('physical DFT voxel-center coordinates; actual source volume, radius 0.15um, boundary clipped; exact plane interpolation weights included' if family=='sbend' else 'unchanged official pixel convention'),
        'is_historical_bitwise_reproduction':False,'role':'formal_sbend_'+rec['split']}
    save(dest/'result.json',result)
    sim.reset_meep()
    print('RESULT '+json.dumps({'geometry_id':gid,'qc':qc,'seconds':result['wall_seconds']},ensure_ascii=False),flush=True)
    if not qc['passed']:raise RuntimeError('QC did not pass; stop expansion')

def smoke():
    t0=time.monotonic();dest=OUT/'smoke';dest.mkdir(exist_ok=False)
    f=1/1.55;w=.5
    ports=[mp.Volume(center=mp.Vector3(x,0),size=mp.Vector3(0,2)) for x in [-2,2]]
    sim=mp.Simulation(cell_size=mp.Vector3(8,4),resolution=20,boundary_layers=[mp.PML(1)],
        geometry=[mp.Block(size=mp.Vector3(mp.inf,w,mp.inf),material=mp.Medium(index=2.4))],
        default_material=mp.Medium(index=1.444),sources=[mp.EigenModeSource(src=mp.GaussianSource(f,fwidth=.1*f),
            center=mp.Vector3(-2.5,0),size=mp.Vector3(0,2),eig_band=1,eig_parity=mp.NO_PARITY,eig_match_freq=True,eig_kpoint=mp.Vector3(1))])
    mons=[sim.add_mode_monitor(f,0,1,mp.ModeRegion(volume=p,direction=mp.X)) for p in ports]
    dft=sim.add_dft_fields([mp.Ez,mp.Hx,mp.Hy],f,0,1,center=mp.Vector3(),size=mp.Vector3(6,2))
    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-6))
    arrays={};modes=[];aa=[];bb=[]
    for i,m in enumerate(mons,1):
        a,b,r=mode_audit(sim,m,f,i,arrays);aa.append(a);bb.append(b);modes.append(r)
    aa=np.array(aa);bb=np.array(bb)
    for n,c in [('Ez',mp.Ez),('Hx',mp.Hx),('Hy',mp.Hy)]:arrays[n]=sim.get_dft_array(dft,c,0).T
    np.savez_compressed(dest/'fields_modes.npz',**arrays)
    trans=aa[1,0]/aa[0,0];refl=aa[0,1]/aa[0,0]
    r={'meep':mp.__version__,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'native_transmission':cjson(trans),'native_reflection':cjson(refl),'transmission_power':float(abs(trans)**2),
        'reflection_amplitude':float(abs(refl)),'Ez_max':float(abs(arrays['Ez']).max()),'modes':modes,
        'EH_transmission':cjson(bb[1,0]/bb[0,0]),'EH_reflection':cjson(bb[0,1]/bb[0,0]),
        'wall_seconds':time.monotonic()-t0,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    r['passed']=bool(r['Ez_max']>1e-6 and .9<r['transmission_power']<1.1 and r['reflection_amplitude']<.1)
    save(dest/'result.json',r);sim.reset_meep()
    print(json.dumps(r,indent=2),flush=True)
    assert r['passed']

if __name__=='__main__':
    mp.verbosity(0)
    np.random.seed(2026)
    if sys.argv[1]=='smoke':smoke()
    else:run_device(sys.argv[1],int(sys.argv[2]))
