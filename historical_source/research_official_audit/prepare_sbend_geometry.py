"""Prepare 16 geometry-only candidates from exact AST-extracted official functions.
No Meep stubs, epsilon rasterization, FDTD fields, or simulated S parameters.
"""
import ast, csv, json
from typing import Optional, List, Dict
import numpy as np
from audit_materials import ROOT, OUT, SRC, save, sha

def extract(rel,names,env):
    path=SRC/rel;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

def main():
    env={'np':np,'Optional':Optional,'List':List,'Dict':Dict}
    # param_ranges.py has only constants and typing-free data; execute read-only.
    exec(compile((SRC/'FDTD/param_ranges.py').read_text(),str(SRC/'FDTD/param_ranges.py'),'exec'),env)
    extract('FDTD/sweep_utils.py',['latin_hypercube'],env)
    extract('FDTD/unified_sweep.py',['sample_params','make_geom_id'],env)
    extract('FDTD/sbend/sbend.py',['euler_sbend_points'],env)
    params=env['sample_params']('sbend',16,seed=2026,resolution=20)
    rng=np.random.default_rng(2026); records=[]
    for i,p in enumerate(params):
        x,y=env['euler_sbend_points'](p['lateral_offset_um'],p['R_min_um'],n_pts=256)
        x-=0.5*(x.min()+x.max());y-=0.5*(y.min()+y.max())
        gid=env['make_geom_id']('sbend',seed=2026*100000+20000+i)
        ip=int(rng.choice([1,2]));w=p['wg_width_um']
        xext=float(np.max(abs(x))+.5*w+.5);yext=float(np.max(abs(y))+.5*w+.5)
        width=float(np.clip(w,.38,.60));code=int(round(float(np.round(width,2))*100))
        table=SRC/f'FDTD/neff_tables/neff_siwire_w{code:03d}_t0220.csv'
        t=np.loadtxt(table,delimiter=',',skiprows=1);ncore=float(np.interp(1.55,t[:,0],t[:,1]))
        span=w+1.0
        records.append({'geometry_id':gid,'candidate_index':i,'device':'sbend','input_port':ip,'params':p,
            'status':'geometry_only_NOT_FDTD','intended_Ez_shape':[160,480],'full_cell_um':[26,10],
            'interior_um':[24,8],'resolution':20,'dx_um':.05,'dy_um':.05,'wavelength_um':1.55,'dpml_um':1,
            'n_core_from_table':ncore,'n_clad':1.444,'neff_table_sha256':sha(table),
            'actual_centerline_offset_um':float(y[-1]-y[0]),'actual_centerline_x_span_um':float(np.ptp(x)),
            'max_extent_with_margin_um':[xext,yext],'fits_official_extent_checks':bool(xext<=12 and yext<=4),
            'port_centers_um':{'1':[-11.5,float(y[0])],'2':[11.5,float(y[-1])]},'port_transverse_span_um':span,
            'actual_source_center_um':[-11.9,float(y[0])] if ip==1 else [11.9,float(y[-1])],
            'official_export_mask_nominal_source_center_um':[-12.0,float(y[0])] if ip==1 else [12.0,float(y[-1])],
            'source_mask_offset_from_actual_um':.1,
            'monitor_span_inside_nonpml_y':bool(min(y[0],y[-1])-span/2>=-4 and max(y[0],y[-1])+span/2<=4)})
    assert len({tuple(r['params'].values()) for r in records})==16
    save('sbend_geometry_candidates_seed2026.json',{'code_archive_commit':'fb752f74fdb3c1cee4db8e6fd6a0bb6cac616f34','seed':2026,
         'status':'16 geometry candidates only; ZERO FDTD samples','records':records})
    summary={'candidate_count':16,'unique_parameter_tuples':16,'fits_extent_checks':sum(r['fits_official_extent_checks'] for r in records),
         'monitor_spans_inside_interior':sum(r['monitor_span_inside_nonpml_y'] for r in records),'generated_fdtd_count':0,
         'source_mask_offset_um':.1,'source_mask_offset_pixels':2,
         'actual_to_requested_offset_ratio_range':[min(r['actual_centerline_offset_um']/r['params']['lateral_offset_um'] for r in records),max(r['actual_centerline_offset_um']/r['params']['lateral_offset_um'] for r in records)],
         'input_port_counts':{str(p):sum(r['input_port']==p for r in records) for p in [1,2]}}
    save('sbend_preflight_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
