"""Read historical wall records, expose missing costs, assemble cost scenarios."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import csv,math,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from predict import R,BASE,OLD,METHODS,SEEDS,read,save,records,weight
B=BASE/'research_sbend_blind_local_phase_2026-11-15'
def csvsave(name,rows):
 with (R/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(v):return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),minimum=float(min(v)),maximum=float(max(v)))
def run():
 resident=read(R/'benchmark_resident.json');cold=read(R/'benchmark_cold.json');prep=read(R/'inputs/preparation.json');formal=read(OLD/'geometry_manifest.json')['records'];cache={r['geometry_id']:r for r in read(OLD/'cache_complete.json')['records']}
 known=[];missing=[]
 for i,r in enumerate(formal):
  path=OLD/'logs'/f'fdtd_{i:03d}_resource.json';result=read(OLD/'data'/r['geometry_id']/'result.json')
  sec=read(path)['wall_seconds'] if path.exists() else result['wall_seconds']
  known.append(dict(component='FDTD_label',split=r['split'],geometry_id=r['geometry_id'],seconds=sec,source=str(path if path.exists() else OLD/'data'/r['geometry_id']/'result.json'),scope='subprocess_wall_with_startup' if path.exists() else 'in_worker_wall_only'))
  c=cache[r['geometry_id']];known.append(dict(component='common_feature_compute',split=r['split'],geometry_id=r['geometry_id'],seconds=c['v1_and_baselines_seconds']+c['v2_seconds'],source=str(OLD/'cache_complete.json'),scope='compute_only; excludes historical input read/cache compression'))
 train=[]
 for seed in SEEDS:
  for m in METHODS:
   s=read(weight(seed,m).with_name('summary.json'));train.append(dict(seed=seed,method=m,seconds=s['train_wall_seconds'],peak_allocated_MiB=s['peak_allocated_MiB'],source=str(weight(seed,m).with_name('summary.json')),scope='historical training loop including validation/checkpointing; excludes setup/data assembly/final evaluation'))
 missing=[dict(component='original_beta_and_scale_calibration',reason='No dedicated historical wall record; original train-only calibration retained; present-day timing replay checked and discarded',quantified_in_measured_subtotal=False),dict(component='historical_training_data_assembly_and_model_setup',reason='Historical training loop timer starts after input loading/model/optimizer setup',quantified_in_measured_subtotal=False),dict(component='historical_feature_IO_and_compression',reason='Historical cache record times geometry computation only',quantified_in_measured_subtotal=False)]
 csvsave('offline_cost_ledger.csv',known);csvsave('training_costs.csv',train);save(R/'unmeasured_costs.json',missing)
 worker_startup=[]
 for q in cold:
  cprep=read(R/'cold_outputs'/(q['method']+'_'+str(q['repeat'])+'_inputs')/'preparation.json');worker_startup.append(max(0.,q['workflow']['total_seconds_after_imports']-q['record']['total_seconds']-cprep['total_seconds']))
 worker_startup_mean=float(np.mean(worker_startup))
 label80=sum(r['seconds'] for r in known if r['component']=='FDTD_label' and r['split'] in ['train','val']);features80=sum(r['seconds'] for r in known if r['component']=='common_feature_compute' and r['split'] in ['train','val']);common=label80+features80
 fdtd=[r['seconds'] for r in read(B/'base_costs.json')];fdtd_mean=float(np.mean(fdtd));rows=[];breakeven=[]
 for m in METHODS:
  cost_train=next(r['seconds'] for r in train if r['seed']==2026 and r['method']==m);offline=common+cost_train
  cr=[r['total_process_seconds'] for r in cold if r['method']==m];warm=next(r for r in resident if r['method']==m and r['mode']=='resident_model_single');wr=[r['total_seconds'] for r in warm['repeats']]
  configs=[('cold_single',stats(cr)['mean'],0.,len(cr)),('resident_single',stats(wr)['mean'],warm['load']['model_load_seconds']+warm['warmup_seconds'],len(wr))]
  for batch in [1,2]:
   d=next(r for r in resident if r['method']==m and r['mode']=='batch' and r['batch_size']==batch);values=[(prep['total_seconds']+worker_startup_mean+r['total_seconds'])/72 for r in d['repeats']];configs.append((f'batch72_b{batch}',float(np.mean(values)),d['load']['model_load_seconds']+d['warmup_seconds'],len(values)))
  for mode,t,setup,n in configs:
   rows.append(dict(method=m,mode=mode,samples=1 if 'single' in mode else 72,repeats=n,query_seconds=t,model_session_setup_seconds=setup,fdtd_reference_seconds=fdtd_mean,online_speed_ratio=fdtd_mean/t,offline_measured_subtotal_seconds=offline,offline_complete=False))
   for overhead in [0.,60.,300.]:
    denominator=fdtd_mean-t;minimum=math.ceil((offline+overhead+setup)/denominator) if denominator>0 else None
    breakeven.append(dict(method=m,mode=mode,scenario='measured_subtotal_lower_bound' if overhead==0 else 'explicit_assumed_missing_overhead',assumed_unmeasured_overhead_seconds=overhead,offline_measured_subtotal_seconds=offline,query_seconds=t,fdtd_seconds=fdtd_mean,minimum_queries=minimum,accuracy_comparable_only_to_resolution20_discrete_reference=True,physical_tolerance_certified=False))
 replay=read(R/'offline_overhead_replay.json');overhead=float(np.mean(replay['calibration_replay_seconds'])+np.mean(replay['training80_assembly_seconds'])+np.mean(replay['model_optimizer_setup_seconds'])+sum(q['total_seconds'] for q in replay['cache80_io']))
 for r in rows:
  r['reconstructed_offline_seconds']=r['offline_measured_subtotal_seconds']+overhead
  r['reconstructed_overhead_seconds']=overhead
  r['batch_preparation_worker_startup_estimate_seconds']=worker_startup_mean if 'batch' in r['mode'] else 0.
  den=fdtd_mean-r['query_seconds'];n=math.ceil((r['reconstructed_offline_seconds']+r['model_session_setup_seconds'])/den) if den>0 else None
  breakeven.append(dict(method=r['method'],mode=r['mode'],scenario='historical_subtotal_plus_current_overhead_replay',assumed_unmeasured_overhead_seconds=overhead,offline_measured_subtotal_seconds=r['offline_measured_subtotal_seconds'],query_seconds=r['query_seconds'],fdtd_seconds=fdtd_mean,minimum_queries=n,accuracy_comparable_only_to_resolution20_discrete_reference=True,physical_tolerance_certified=False))
 csvsave('online_cost_summary.csv',rows);csvsave('break_even_scenarios.csv',breakeven)
 prep_parts={k:stats([r[k] for r in prep['costs']]) for k in ['material_init_seconds','feature_seconds','export_seconds','total_seconds']}
 external=sum(r['seconds'] for r in known if r['component']=='FDTD_label' and r['split']=='test')+sum(fdtd)
 measured_research=dict(all12_model_training_seconds=sum(r['seconds'] for r in train),formal180_label_seconds=sum(r['seconds'] for r in known if r['component']=='FDTD_label'),formal180_feature_compute_seconds=sum(r['seconds'] for r in known if r['component']=='common_feature_compute'),formal100_and_blind72_evaluation_label_seconds=external,blind72_online_feature_seconds=sum(r['seconds'] for r in read(B/'feature_costs.json')),blind12_prediction_workflows_seconds=sum(r['seconds'] for r in read(B/'inference_costs.json')),scope='partial ledger of identified study costs; not total research cost; omits early development, abandoned models, Y, grid diagnoses, new delivery validation/benchmarking')
 save(R/'cost_analysis.json',dict(training_validation_label_seconds=label80,training_validation_feature_compute_seconds=features80,shared_offline_measured_subtotal_seconds=common,fdtd_baseline=stats(fdtd),preprocessing=prep_parts,preparation_worker_startup_estimate=stats(worker_startup),unmeasured_costs=missing,reconstructed_missing_overhead_seconds=overhead,overhead_replay=replay,online=rows,break_even=breakeven,identified_research_costs=measured_research,notes=['All wall time, not energy or currency; CPU FDTD versus CPU+GPU surrogate on named hardware','Batch throughput includes geometry initialization, all common features/four carriers, input export/load, transfer/inference/output export; amortizes preprocessing process imports across72','Resident single includes fresh geometry worker process per query; GPU net retained','Cold single includes full Python/GPU/geometry-worker startup and model load','Common feature information includes Path v2 in every model, required for exact historical replay','FDTD archive exports H/port audits beyond the Ez-only deliverable; absolute acceleration ratio is pipeline-specific, not an optimized solver comparison','Missing calibration/setup/cache IO must be added to subtotal; 0/60/300 seconds are sensitivity assumptions, never measured totals','Accuracy rates are ex post on saved truth; no online error detector or guaranteed FDTD fallback savings']));
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42});fig,axs=plt.subplots(1,3,figsize=(14,4.2),layout='constrained');colors=['#818181','#d28e23','#137d99','#7853a0'];labels=['Linear','ArcOnly','LocalTangent','Path v2']
 for j,m in enumerate(METHODS):
  vals=[next(r['query_seconds'] for r in rows if r['method']==m and r['mode']==mode) for mode in ['cold_single','resident_single','batch72_b2']];axs[0].plot(range(3),vals,'o-',label=labels[j],color=colors[j])
 axs[0].axhline(fdtd_mean,color='black',ls='--',label='Archived FDTD pipeline');axs[0].set_xticks(range(3),['Cold single','Resident single','Batch72 / query']);axs[0].set_ylabel('Wall seconds per geometry');axs[0].set_ylim(bottom=0);axs[0].legend(fontsize=7);axs[0].set_title('Measured query pipeline (seed2026)')
 ns=np.linspace(0,1500,301);lt=[r for r in rows if r['method']=='local_tangent' and r['mode']=='batch72_b2'][0];axs[1].plot(ns,ns*fdtd_mean/60,'k--',label='FDTD');axs[1].plot(ns,(lt['offline_measured_subtotal_seconds']+lt['model_session_setup_seconds']+ns*lt['query_seconds'])/60,color='#137d99',label='LT measured subtotal');axs[1].fill_between(ns,(lt['offline_measured_subtotal_seconds']+lt['model_session_setup_seconds']+ns*lt['query_seconds'])/60,(lt['offline_measured_subtotal_seconds']+lt['model_session_setup_seconds']+300+ns*lt['query_seconds'])/60,color='#137d99',alpha=.18,label='+0 to300s assumed overhead');axs[1].set_xlabel('Number of new geometry queries');axs[1].set_ylabel('Accumulated wall minutes');axs[1].legend(fontsize=7);axs[1].set_title('Illustrative repeated-batch amortization')
 rates=[]
 with (R/'illustrative_tolerance_rates.csv').open(encoding='utf-8-sig') as f:rates=list(csv.DictReader(f))
 for j,m in enumerate(METHODS):
  vals=[np.mean([float(r['fraction']) for r in rates if r['method']==m and r['criterion']=='whole_rel_l2' and float(r['threshold'])==t]) for t in [.1,.15,.2]];axs[2].plot([.1,.15,.2],vals,'o-',color=colors[j],label=labels[j])
 axs[2].set_xlabel('Illustrative whole-field RelL2 tolerance');axs[2].set_ylabel('Observed fraction within tolerance');axs[2].set_ylim(0,1.02);axs[2].legend(fontsize=7);axs[2].set_title('72 cases; mean of three seed rates')
 for ax in axs:ax.grid(alpha=.2)
 fig.suptitle('Resolution20 discrete-reference approximation | no continuous-physics accuracy guarantee',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(R/f'cost_and_accuracy.{ext}',dpi=180)
 plt.close(fig);save(R/'cost_analysis_complete.json',dict(passed=True,full_offline_cost_measured=False,unmeasured_components=len(missing)))
if __name__=='__main__':run()


