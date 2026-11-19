"""Read-only provenance/schema audit. Outputs only in this new audit directory."""
from pathlib import Path
import ast, collections, hashlib, json, os, subprocess, sys, zipfile
import numpy as np

ROOT = Path('/mnt/d/game')
NEW = ROOT / 'PIC-Flow-新研究资料'
OUT = ROOT / 'research_official_audit'
SRC = next(p.parent for p in NEW.rglob('README.md') if (p.parent/'FDTD/unified_sweep.py').exists())

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def historic():
    paths=[]
    for folder in ROOT.glob('research_*'):
        if folder == OUT: continue
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.pt','.json','.md'))
    paths += [ROOT/'picflow_source/shards/index.json', ROOT/'研究交接_给下一位Agent.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}

def main():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'historical_before.json').exists(): save('historical_before.json',historic())
    inventory={str(p.relative_to(NEW)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(NEW.rglob('*')) if p.is_file()}
    save('official_file_hashes.json',inventory)
    zpath=NEW/'PIC-Flow-main.zip'
    with zipfile.ZipFile(zpath) as z:
        prefix=next(n[:-len('FDTD/unified_sweep.py')] for n in z.namelist() if n.endswith('FDTD/unified_sweep.py'))
        mismatch=[]
        count=0
        for n in z.namelist():
            if n.endswith('/') or not n.startswith(prefix):continue
            p=SRC/n[len(prefix):]
            count+=1
            if not p.exists() or hashlib.sha256(z.read(n)).hexdigest()!=sha(p):mismatch.append(n)
        archive={'path':str(zpath),'sha256':sha(zpath),'zip_comment':z.comment.decode('utf-8',errors='replace'),
                 'crc_first_bad_member':z.testzip(),'files_checked':count,'extraction_mismatches':mismatch}
    oldidx=ROOT/'picflow_source/shards/index.json'
    idx=json.loads((NEW/'index.json').read_text())
    old=json.loads(oldidx.read_text())
    geos=[json.loads(line) for line in (NEW/'geometries.jsonl').read_text().splitlines() if line.strip()]
    gid={g['geometry_id']:g for g in geos}
    available={p.name for p in (ROOT/'picflow_source/shards').glob('shard_*.npz')}
    index_audit={'new_index_sha256':sha(NEW/'index.json'),'old_index_sha256':sha(oldidx),
                 'byte_identical':sha(NEW/'index.json')==sha(oldidx),'parsed_identical':idx==old,
                 'index_rows':len(idx),'unique_geometry_ids':len({r['geometry_id'] for r in idx}),
                 'geometries_rows':len(geos),'geometries_unique_ids':len(gid),
                 'index_geometry_metadata_missing':sum(r['geometry_id'] not in gid for r in idx),
                 'device_split_counts':dict(collections.Counter(f"{r['device']}/{r['split']}/{r['augment']}" for r in idx)),
                 'local_shards':len(available),'available_counts':dict(collections.Counter(f"{r['device']}/{r['split']}" for r in idx if r['shard'] in available))}
    # Use only already used original-80 geometries for numerical audits.
    with np.load(ROOT/'picflow_ybranch_smoke_80.npz',allow_pickle=False) as d:
        smoke_keys={k:{'shape':list(d[k].shape),'dtype':str(d[k].dtype)} for k in d.files}
        ids=np.asarray(d['geometry_id']).astype(str).tolist()
    selected=[r for r in idx if r['geometry_id'] in set(ids) and r['augment']=='orig' and r['shard'] in available]
    schema=[]
    for row in selected:
        with np.load(ROOT/'picflow_source/shards'/row['shard'],allow_pickle=False) as d:
            pre=f"s{row['slot']}/"
            keys=[k[len(pre):] for k in d.files if k.startswith(pre)]
            val=lambda k:d[pre+k]
            params={k[7:]:float(val(k)) for k in keys if k.startswith('params/')}
            meta=gid[row['geometry_id']]
            s_names=sorted(k[:-5] for k in keys if k.startswith('sparams/') and k.endswith('_real'))
            port=int(val('input_port'))
            pm=val('port_masks'); src=val('src_mask'); eps=val('eps')
            spans=[]
            for pid,m in zip(val('port_ids'),pm):
                yy,xx=np.where(m>0)
                spans.append({'port':int(pid),'x_pixel_bounds':[int(xx.min()),int(xx.max())],
                              'y_pixel_bounds':[int(yy.min()),int(yy.max())],
                              'center_pixel':[float(np.mean(xx)),float(np.mean(yy))]})
            yy,xx=np.where(src>0)
            schema.append({**row,'input_port':port,'keys':keys,'params':params,
                'geometries_parameter_match':all(abs(params[k]-meta[k])<1e-6 for k in params),
                'geometries_input_port_match':int(meta['input_port'])==port,
                'eps_shape':list(eps.shape),'eps_min':float(eps.min()),'eps_max':float(eps.max()),
                'Ez_dtype':str(val('Ez_real').dtype),'finite_fields':bool(np.isfinite(val('Ez_real')).all() and np.isfinite(val('Ez_imag')).all()),
                'grid':{k:float(val(k)) for k in ['nx','ny','Lx_um','Ly_um','dx_um','dy_um','resolution','wavelength_um']},
                'port_ids':val('port_ids').tolist(),'port_masks':spans,'source_center_pixel':[float(xx.mean()),float(yy.mean())],
                'sparam_names':s_names,'sparam_power_sum':float(sum(abs(complex(float(val(k+'_real')),float(val(k+'_imag'))))**2 for k in s_names)),
                'has_H':any(k.startswith(('Hx','Hy')) for k in keys)})
    save('sample_schema_audit.json',schema)
    save('used80_index_records.json',selected)
    runtime=[]
    prefixes=[Path('/usr'),Path('/home/cudauser/miniconda3')]
    prefixes+=list(Path('/home/cudauser/.venvs').glob('*'))+list(Path('/home/cudauser/miniconda3/envs').glob('*'))
    probe="import importlib.util,json,sys; s=importlib.util.find_spec('meep'); print(json.dumps({'python':sys.executable,'version':sys.version,'meep_spec':str(s)}))"
    for prefix in prefixes:
        py=prefix/'bin/python'
        if not py.exists():py=prefix/'bin/python3'
        if not py.exists():continue
        try:
            p=subprocess.run([str(py),'-c',probe],capture_output=True,text=True,timeout=20)
            runtime.append({'prefix':str(prefix),'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()})
        except Exception as e: runtime.append({'prefix':str(prefix),'error':repr(e)})
    save('environment_probe.json',runtime)
    funcs={}
    for rel in ['FDTD/ybranch/ybranch.py','FDTD/sbend/sbend.py','FDTD/unified_sweep.py','FDTD/utils.py','FDTD/pack_dataset.py','FDTD/param_ranges.py','Model/dataset.py']:
        tree=ast.parse((SRC/rel).read_text(encoding='utf-8-sig'))
        funcs[rel]=[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    save('source_symbols.json',funcs)
    result={'source_root':str(SRC),'archive':archive,'index':index_audit,'smoke_npz_keys':smoke_keys,
            'audited_historical_samples':len(schema),'all_metadata_params_match':all(r['geometries_parameter_match'] for r in schema),
            'all_metadata_ports_match':all(r['geometries_input_port_match'] for r in schema),
            'meep_found':any('meep_spec' in r.get('stdout','') and 'None' not in r['stdout'] for r in runtime),
            'generated_sbend_fdtd_samples':0}
    save('materials_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
