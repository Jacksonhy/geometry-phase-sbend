"""Extra source attribution tables; all diagnostics, never prediction correction."""
import numpy as np
from experiment import R,read,save
from analyze import csvsave
b=read(R/'beta.json');lookup={r['job_id']:r for r in b};cfg=read(R/'config.json');d=read(R/'decomposition.json');p=read(R/'port_reference.json');rows=[]
for r in cfg['diagnostic_records']:
 gid=r['geometry_id'];w=r['params']['wg_width_um']
 for lo,hi in [(20,30),(30,40),(40,60)]:
  a=next(v for v in p if v['geometry_id']==gid and v['resolution']==lo);z=next(v for v in p if v['geometry_id']==gid and v['resolution']==hi);bl=lookup[f'straight_{gid}_r{lo}'];bh=lookup[f'straight_{gid}_r{hi}'];db=bh['beta_FDTD']-bl['beta_FDTD'];dm=bh['beta_MPB']-bl['beta_MPB'];pair=next(v for v in d if v['geometry_id']==gid and v['low_resolution']==lo)
  wrap=lambda x:float(np.degrees(np.angle(np.exp(1j*np.radians(x)))))
  extra=wrap(z['extra_physical_phase_deg']-a['extra_physical_phase_deg']);straight=wrap(z['straight_physical_phase_deg']-a['straight_physical_phase_deg']);total=wrap(z['physical_t_phase_deg']-a['physical_t_phase_deg']);extra_pred=np.degrees(db*a['delta_length_um']);centerlo=lookup[f'centered_w{w:.3f}_r{lo}'];centerhi=lookup[f'centered_w{w:.3f}_r{hi}'];rows.append(dict(geometry_id=gid,width_um=w,angle_stratum=r['angle_stratum'],input_port=r['input_port'],low_resolution=lo,high_resolution=hi,delta_beta_FDTD=db,delta_beta_MPB=dm,straight_port_phase_change_deg=straight,Sbend_port_phase_change_deg=total,bend_extra_phase_change_deg=extra,predicted_common_beta23_deg=float(np.degrees(db*23)),common_beta23_residual_deg=wrap(straight-np.degrees(db*23)),predicted_beta_extra_length_deg=float(extra_pred),bend_extra_after_betaDeltaL_deg=wrap(extra-extra_pred),beta_input_shift_lo_minus_center=bl['beta_FDTD']-centerlo['beta_FDTD'],beta_input_shift_hi_minus_center=bh['beta_FDTD']-centerhi['beta_FDTD'],bend_local_original_WMAE_deg=pair['bend_phase_change_wmae_deg'],bend_local_after_beta_WMAE_deg=pair['residual_after_delta_beta_wmae_deg']))
save(R/'attribution_tables.json',rows);csvsave(R/'attribution_tables.csv',rows)
# Compare temporal Courant sensitivity with known spatial-dispersion grid shift.
courant=[]
for w in cfg['widths_um']:
 a=lookup[f'centered_w{w:.3f}_r40'];z=lookup[f'courant025_w{w:.3f}_r40'];courant.append(dict(width_um=w,beta_Courant05=a['beta_FDTD'],beta_Courant025=z['beta_FDTD'],delta_beta=z['beta_FDTD']-a['beta_FDTD'],phase23_change_deg=float(np.degrees((z['beta_FDTD']-a['beta_FDTD'])*23)),analytic_beta=a['beta_analytic_slab']))
save(R/'courant_sensitivity.json',courant);print('ATTRIBUTION_TABLES_READY',len(rows),flush=True)
