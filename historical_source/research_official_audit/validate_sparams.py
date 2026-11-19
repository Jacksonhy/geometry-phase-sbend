"""CPU-only truth-field audit, no training, fitting, or query phase calibration."""
import ast, csv, json, math, sys, types
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh_tridiagonal
from audit_materials import ROOT, NEW, OUT, SRC, save, sha

torch.set_num_threads(4)

def load_readonly(name, rel):
    m=types.ModuleType(name)
    m.__file__=str(SRC/rel)
    exec(compile((SRC/rel).read_text(encoding='utf-8-sig'),m.__file__,'exec'),m.__dict__)
    return m

modal=load_readonly('official_modal','Model/modal_sparams.py')
simple=load_readonly('official_simple','Model/sparams_loss.py')

def fd_directional_proxy(E,eps,masks,port_ids,ip,wl,dx):
    """Approximate scalar mode + normal derivative; NOT Meep/MPB reproduction.

    Dirichlet 1D eigensystem on transverse mask bounds padded 0.5um,
    second-order d/dx, positive real modal gauge, x-directed ports only.
    """
    coeffs=[]; betas=[]
    for mask in masks:
        yy,xx=np.where(mask>0.5)
        x=int(xx.mean()); lo=max(0,int(yy.min())-10); hi=min(E.shape[0],int(yy.max())+11)
        ep=eps[lo:hi,x].astype(np.float64)
        diag=(2*np.pi/wl)**2*ep-2/dx**2
        vals,vecs=eigh_tridiagonal(diag,np.full(len(ep)-1,1/dx**2),select='i',select_range=(len(ep)-1,len(ep)-1))
        beta=float(np.sqrt(vals[0]));u=vecs[:,0]
        if u.sum()<0:u=-u
        u=u/np.sqrt(np.sum(u*u)*dx)
        e=np.sum(u*E[lo:hi,x])*dx
        de=np.sum(u*(E[lo:hi,x+1]-E[lo:hi,x-1])/(2*dx))*dx
        coeffs.append((0.5*(e+de/(1j*beta)),0.5*(e-de/(1j*beta))))
        betas.append(beta)
    ii=list(port_ids).index(ip)
    ain=coeffs[ii][0 if ip==1 else 1]*np.sqrt(betas[ii])
    return np.array([c[1 if int(p)==1 else 0]*np.sqrt(b)/ain for p,c,b in zip(port_ids,coeffs,betas)])

def metrics(rows):
    t=np.array([complex(r['official_real'],r['official_imag']) for r in rows])
    p=np.array([complex(r['estimate_real'],r['estimate_imag']) for r in rows])
    gate=np.abs(t)>=0.05
    return {'coefficients':len(rows),'complex_rmse':float(np.sqrt(np.mean(np.abs(p-t)**2))),
            'complex_mae':float(np.mean(np.abs(p-t))),
            'magnitude_mae':float(np.mean(np.abs(np.abs(p)-np.abs(t)))),
            'power_mae':float(np.mean(np.abs(np.abs(p)**2-np.abs(t)**2))),
            'phase_abs_mean_deg_true_mag_ge_005':float(np.mean(np.abs(np.angle(p[gate]*t[gate].conj())))*180/np.pi),
            'phase_gated_count':int(gate.sum())}

def main():
    records=json.loads((OUT/'used80_index_records.json').read_text())
    result=[]; residuals=[]
    for r in records:
        with np.load(ROOT/'picflow_source/shards'/r['shard'],allow_pickle=False) as d:
            pre=f"s{r['slot']}/"; get=lambda k:d[pre+k]
            E=get('Ez_real').astype(np.float64)+1j*get('Ez_imag').astype(np.float64)
            ep=get('eps').astype(np.float64); pm=get('port_masks').astype(np.float64)
            ids=get('port_ids').astype(int);ip=int(get('input_port'));ii=ids.tolist().index(ip)
            wl=float(get('wavelength_um'));dx=float(get('dx_um'))
            Et=torch.from_numpy(E);ept=torch.from_numpy(ep);pmt=torch.from_numpy(pm)
            with torch.no_grad():
                m=modal.extract_sparams_modal(Et,pmt,ept,wl,dx,in_port_idx=ii,port_ids=torch.from_numpy(ids)).numpy()
                s=simple.extract_sparams(Et[None],pmt,in_port_idx=ii,port_ids=torch.from_numpy(ids)).numpy()[0]
                # Test the named "solve_1d_mode_fdfd" against its stated Helmholtz equation.
                u,neff,coords=modal.compute_mode_profile_from_eps(ept,pmt[0],wl,dx,'x')
                x=int(torch.nonzero(pmt[0]>.5)[:,1].float().mean())
                profile=ept[coords.long(),x].numpy();un=u.numpy();k0=2*np.pi/wl
                residual=(un[:-2]-2*un[1:-1]+un[2:])/dx**2+(k0**2*profile[1:-1]-(k0*neff)**2)*un[1:-1]
                residuals.append(float(np.linalg.norm(residual)/(k0**2*np.max(profile)*np.linalg.norm(un[1:-1]))))
            f=fd_directional_proxy(E,ep,pm,ids,ip,wl,dx)
            for method,values in [('official_code_modal_proxy',m),('official_code_mask_average_proxy',s),('audit_fd_directional_proxy',f)]:
                for pid,est in zip(ids,values):
                    true=complex(float(get(f'sparams/S{pid}{ip}_real')),float(get(f'sparams/S{pid}{ip}_imag')))
                    result.append({'geometry_id':r['geometry_id'],'input_port':ip,'output_port':int(pid),'method':method,
                                   'is_reflection':bool(pid==ip),'official_real':true.real,'official_imag':true.imag,
                                   'estimate_real':float(est.real),'estimate_imag':float(est.imag)})
    with (OUT/'sparam_truth_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    summary={}
    for method in sorted({r['method'] for r in result}):
        sub=[r for r in result if r['method']==method]
        summary[method]={name:metrics([r for r in sub if pred(r)]) for name,pred in {
            'all':lambda r:True,'reflection':lambda r:r['is_reflection'],
            'noninput':lambda r:not r['is_reflection'],
            'input1_noninput':lambda r:r['input_port']==1 and not r['is_reflection'],
            'input2_noninput':lambda r:r['input_port']==2 and not r['is_reflection'],
            'input3_noninput':lambda r:r['input_port']==3 and not r['is_reflection']}.items()}
    save('sparam_validation_summary.json',{'sample_count':len(records),'data_role':'historical original80; posthoc extractor validation, not blind model evaluation',
         'no_parameters_fitted_to_sparams':True,'official_proxy_mode_equation_relative_residual':{'mean':float(np.mean(residuals)),'max':float(np.max(residuals))},
         'methods':summary,'strict_meep_reproduction_validated':False})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
