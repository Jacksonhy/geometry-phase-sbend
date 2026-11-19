"""Consistency of fixed incident-reference normalization in saved grid tables."""
from experiment import R,read,save
import csv

def load(name):
 with (R/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
rows=load('grid_per_geometry.csv');keys=['kind','truth_resolution','seed','method','geometry_id','region','gate','eta'];lookup={(tuple(r[k] for k in keys),r['convention']):r for r in rows};common=[];phase=[]
for r in rows:
 if r['convention']!='raw':continue
 other=lookup[tuple(r[k] for k in keys),'incident_normalized']
 if r['kind']=='pair':
  phase.append(abs(float(r['normal_relative_phase_wmae_deg'])-float(other['normal_relative_phase_wmae_deg'])))
 elif r['truth_resolution']=='20':
  for m in ['phase_wmae_deg','complex_rel_l2']:common.append(abs(float(r[m])-float(other[m])))
assert max(common)<1e-10 and max(phase)<1e-10
qs=read(R/'grid_incoming_reference.json');save(R/'normalization_verification.json',dict(passed=True,res20_raw_vs_normalized_max_metric_difference=max(common),normal_pair_phase_common_phase_invariance_max_difference_deg=max(phase),incoming_EH_vs_native_max_phase_difference_deg=max(abs(r['EH_vs_native_phase_deg']) for r in qs),incoming_EH_vs_native_max_complex_relative_difference=max(r['EH_vs_native_amplitude_relative'] for r in qs),input_reference_not_fitted_to_bend_truth=True))
print('NORMALIZATION_VERIFIED',max(common),max(phase),flush=True)
