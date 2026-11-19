"""Verify historical hashes again and freeze the independent delivery."""
from analyze_phase import *
import re,scipy

def main():
 before=read(OUT/'input_hashes_before.json')
 changed=[name for name,digest in before.items() if sha(ROOT/name)!=digest]
 assert not changed,changed
 frozen_common.verify_freeze()
 assert sha(OUT/'analyze_phase.py')==read(OUT/'diagnostic_freeze.json')['script_sha256']
 assert sha(OUT/'analysis_config.json')==read(OUT/'diagnostic_freeze.json')['config_sha256']
 with (OUT/'spectral_per_geometry.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 assert len(rows)==9600
 assert {r['subset'] for r in rows}=={'train64','development16'}
 report=OUT/'几何相位分解_理论频谱与曲率模态可行性报告.md'
 content=report.read_text(encoding='utf-8')
 assert all(f'## {i}.' in content for i in [1,2,3,4,5,6,8])
 write('delivery_verification.json',{'historical_files_checked':len(before),'historical_files_changed':changed,
  'original_analysis_script_and_config_unchanged':True,'formal_code_freeze_passed':True,'per_geometry_spectrum_rows':len(rows),
  'dataset_geometry_counts':{s:len({r['geometry_id'] for r in rows if r['subset']==s}) for s in ['development16','train64']},
  'mean_spectrum_files':len(list((OUT/'mean_spectra').glob('*.npz'))),'figure_png_files':len(list((OUT/'figures').glob('*.png'))),
  'report_characters':len(content),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
  'FDTD_runs':0,'network_training_or_inference_runs':0,'new_carrier_candidates':0})
 # First create manifest so report's forward link resolves during link checking.
 write('outputs_manifest.json',{str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='outputs_manifest.json'})
 missing=[]
 for raw in re.findall(r'\]\((D:/[^)]+)\)',content):
  path=re.sub(r':\d+$','',raw).replace('D:/','/mnt/d/')
  if not Path(path).exists():missing.append(raw)
 assert not missing,missing
 print(json.dumps({'verified_historical':len(before),'spectrum_rows':len(rows),'files':len(read(OUT/'outputs_manifest.json')),'all_report_local_links_resolve':True},indent=2),flush=True)
if __name__=='__main__':main()
