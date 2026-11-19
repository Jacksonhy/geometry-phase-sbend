"""Final finite audit and packaging; no model or solver execution."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,subprocess,csv,math,zipfile
from pathlib import Path
import numpy as np
from predict import R,read,save,digest
from audit import verify

def finish():
 verify()
 for name in ['inputs_verification.json','prediction_replay_verification.json','table_reconstruction_verification.json','independent_delivery_verification.json','benchmarks_complete.json','offline_overhead_replay.json','cost_analysis_complete.json','integrated_tables_complete.json','figures_complete.json']:
  assert read(R/name)['passed'],name
 a=read(R/'inputs_verification.json');assert a['n']==72
 checks=read(R/'prediction_replay_verification.json')['checks'];assert len(checks)==12 and all(q['bitwise_equal'] for q in checks)
 with np.load(R/'final_cli_example.npz') as x,np.load(R/'cold_outputs/local_tangent_0.npz') as y:
  assert np.array_equal(x['prediction'],y['prediction']);assert np.array_equal(x['x_um'],y['x_um']) and np.array_equal(x['y_um'],y['y_um'])
 # Wrong prepared input must fail before model loading/producing output.
 r=read(R/'example_geometry.json');r['input_port']=3-r['input_port'];save(R/'mismatched_geometry_test.json',r)
 command=[sys.executable,'-B',str(R/'predict.py'),'--geometry-json',str(R/'mismatched_geometry_test.json'),'--prepared-inputs',str(R/'smoke_inputs'),'--output',str(R/'must_not_exist.npz')]
 p=subprocess.run(command,capture_output=True,text=True);assert p.returncode!=0 and 'Prepared inputs do not match' in p.stderr;assert not (R/'must_not_exist.npz').exists();(R/'logs/final_contract_rejection.log').write_text(p.stdout+p.stderr,encoding='utf-8')
 # Independent arithmetic from raw historical ledger + raw training and timing summaries.
 ledger=[]
 with (R/'offline_cost_ledger.csv').open(encoding='utf-8-sig') as f:ledger=list(csv.DictReader(f))
 cost=read(R/'cost_analysis.json');label=math.fsum(float(r['seconds']) for r in ledger if r['component']=='FDTD_label' and r['split'] in ['train','val']);assert abs(label-cost['training_validation_label_seconds'])<1e-9
 for q in cost['break_even']:
  if q['scenario']!='historical_subtotal_plus_current_overhead_replay':continue
  r=next(a for a in cost['online'] if a['method']==q['method'] and a['mode']==q['mode']);expected=math.ceil((r['reconstructed_offline_seconds']+r['model_session_setup_seconds'])/(r['fdtd_reference_seconds']-r['query_seconds']));assert expected==q['minimum_queries']
 assert digest(R/'predict_benchmarked.py')==read(R/'freeze.json')['predict_initial_sha256']
 save(R/'final_verification.json',dict(passed=True,geometries_exact=72,model_replays_bitwise_exact=12,final_CLI_matches_benchmarked_prediction=True,prepared_input_mismatch_rejected=True,offline_ledger_and_break_even_arithmetic_checked=True,predict_initial_version_preserved=True,final_input_contract_seconds=read(R/'final_cli_example.json')['input_contract_check_seconds'],training_runs=0,FDTD_runs=0,deployed_calibration_changes=0,historical_files_unchanged=577))
 save(R/'completion.json',dict(passed=True,unix=time.time(),elapsed_since_freeze_seconds=time.time()-read(R/'freeze.json')['unix'],requested_deliverables=['geometry-only prediction script and tested CLI','end-to-end costs and break-even analysis','72geometry3seed main tables and full-field galleries'],report_sha256=digest(R/'REPORT.md'),predict_sha256=digest(R/'predict.py'),training_runs=0,FDTD_runs=0,new_truth_fields=0,original_predictions_unchanged=True))
 excluded={'PACKAGE_MANIFEST.json','archive_complete.json','release_bundle.zip'}
 paths=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.md','.csv','.json','.png','.pdf','.npz'] and p.name not in excluded and not p.name.startswith(('invalid_','mismatched_'))]
 manifest=dict(files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)],excludes='weights, source dependencies outside this folder, large input/replay/benchmark arrays and logs; current environment is required',final_CLI_example_included=True)
 save(R/'PACKAGE_MANIFEST.json',manifest)
 with zipfile.ZipFile(R/'release_bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(paths)+[R/'PACKAGE_MANIFEST.json']:z.write(p,p.name)
 with zipfile.ZipFile(R/'release_bundle.zip') as z:assert z.testzip() is None
 save(R/'archive_complete.json',dict(passed=True,archive_sha256=digest(R/'release_bundle.zip'),bytes=(R/'release_bundle.zip').stat().st_size,files=len(paths)+1))
 print('DELIVERY_COMPLETE',len(paths)+1,flush=True)
if __name__=='__main__':finish()
