"""Independent scalar invariants and provenance checks on already used data."""
import ast, json
import numpy as np
from audit_materials import ROOT, OUT, SRC, save
from validate_sparams import fd_directional_proxy

def main():
    records=json.loads((OUT/'used80_index_records.json').read_text())
    raw_diffs=[];anchor_angles=[];core_eps_diffs=[];core_eps_meta_diffs=[];ports=[];power=[]
    metas={r['geometry_id']:r for r in map(json.loads,(ROOT/'PIC-Flow-新研究资料/geometries.jsonl').read_text().splitlines())}
    with np.load(ROOT/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as smoke:
        ids=smoke['geometry_id'].astype(str).tolist()
        smoke_real=smoke['Ez_real'];smoke_imag=smoke['Ez_imag']
        for r in records:
            with np.load(ROOT/'picflow_source/shards'/r['shard'],allow_pickle=False) as d:
                pre=f"s{r['slot']}/";v=lambda k:d[pre+k]
                E=v('Ez_real').astype(np.float64)+1j*v('Ez_imag').astype(np.float64)
                i=ids.index(r['geometry_id'])
                raw_diffs.append(float(np.max(np.abs(E-(smoke_real[i].astype(np.float64)+1j*smoke_imag[i].astype(np.float64))))))
                weight=(v('src_mask')>0)*(v('eps')>3)
                anchor_angles.append(float(np.angle(np.sum(weight*np.abs(E)*E))))
                width=float(np.clip(float(v('params/wg_width_um')),.38,.60))
                code=int(round(float(np.round(width,2))*100))
                table=np.loadtxt(SRC/f'FDTD/neff_tables/neff_siwire_w{code:03d}_t0220.csv',delimiter=',',skiprows=1)
                neff=float(np.interp(float(v('wavelength_um')),table[:,0],table[:,1]))
                core_eps_diffs.append(abs(neff**2-float(v('eps').max())))
                width_meta=float(np.clip(metas[r['geometry_id']]['wg_width_um'],.38,.60))
                code_meta=int(round(float(np.round(width_meta,2))*100))
                table_meta=np.loadtxt(SRC/f'FDTD/neff_tables/neff_siwire_w{code_meta:03d}_t0220.csv',delimiter=',',skiprows=1)
                neff_meta=float(np.interp(1.55,table_meta[:,0],table_meta[:,1]))
                core_eps_meta_diffs.append(abs(neff_meta**2-float(v('eps').max())))
                ip=int(v('input_port'));s=np.array([complex(float(v(f'sparams/S{p}{ip}_real')),float(v(f'sparams/S{p}{ip}_imag'))) for p in [1,2,3]])
                power.append(float(np.sum(abs(s)**2)));ports.append(ip)
    # Discrete transverse eigenmode with continuum longitudinal plane waves.
    # Verify signs and power weighting without official S labels or any fitted parameter.
    from scipy.linalg import eigh_tridiagonal
    dy=.05;wl=1.55;H=80;W=100;xref=50; k0=2*np.pi/wl
    eps1=np.full(H,1.444**2);eps1[35:45]=2.4**2
    lam,vec=eigh_tridiagonal(k0*k0*eps1-2/dy**2,np.full(H-1,1/dy**2),select='i',select_range=(H-1,H-1))
    beta=np.sqrt(lam[0]);u=vec[:,0];u*=np.sign(u.sum());u/=np.sqrt(np.sum(u*u)*dy)
    x=(np.arange(W)-xref)*dy
    mask=np.zeros((H,W));mask[:,xref]=1
    # Forward-only field: port2 outgoing/port1 incoming = +1 at the same reference x.
    synthetic=fd_directional_proxy(u[:,None]*np.exp(1j*beta*x)[None,:],np.tile(eps1[:,None],(1,W)),np.stack([mask,mask]),[1,2],1,wl,dy)
    mixed_a=1.2+.3j;mixed_b=.08-.04j
    mixed=u[:,None]*(mixed_a*np.exp(1j*beta*x)+mixed_b*np.exp(-1j*beta*x))[None,:]
    mixed_proxy=fd_directional_proxy(mixed,np.tile(eps1[:,None],(1,W)),np.stack([mask,mask]),[1,2],1,wl,dy)
    out={'historical_raw_field_vs_smoke_max_abs_diff':max(raw_diffs),
         'source_anchor_phase_abs_mean_rad':float(np.mean(abs(np.array(anchor_angles)))),
         'source_anchor_phase_abs_max_rad':float(np.max(abs(np.array(anchor_angles)))),
         'core_eps_vs_downloaded_neff_table_max_abs_diff':max(core_eps_diffs),
         'core_eps_vs_geometries_jsonl_table_max_abs_diff':max(core_eps_meta_diffs),
         'input_port_counts':{str(p):ports.count(p) for p in [1,2,3]},
         'guided_sparam_power_sum':{'min':min(power),'median':float(np.median(power)),'max':max(power)},
         'synthetic_forward_only_reflection_artifact_abs':float(abs(synthetic[0])),
         'synthetic_forward_only_transmission_abs_error':float(abs(synthetic[1]-1)),
         'synthetic_mixed_reflection_expected':[float((mixed_b/mixed_a).real),float((mixed_b/mixed_a).imag)],
         'synthetic_mixed_reflection_estimate':[float(mixed_proxy[0].real),float(mixed_proxy[0].imag)],
         'synthetic_mixed_reflection_abs_error':float(abs(mixed_proxy[0]-mixed_b/mixed_a)),
         'synthetic_expected_central_difference_reflection_floor':float(abs((1-np.sin(beta*dy)/(beta*dy))/(1+np.sin(beta*dy)/(beta*dy))))}
    assert abs(abs(synthetic[0])-out['synthetic_expected_central_difference_reflection_floor'])<1e-12
    save('numeric_invariants.json',out)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
