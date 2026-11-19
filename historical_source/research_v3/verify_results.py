"""Scientific metric identities and provenance/replay checks for this result bundle."""
import hashlib
import json
import numpy as np
from experiment import ROOT, metric_arrays, geometry_features, write_json

target=np.array([[[1+2j,3-1j],[2+1j,-1-2j]]],dtype=np.complex64)
same=metric_arrays(target,target)
assert np.allclose(same['rel_l2'],0)
assert np.allclose(same['intensity_ratio'],1)
assert np.allclose(metric_arrays(np.zeros_like(target),target)['rel_l2'],1)
rotated=metric_arrays(1j*target,target)
assert np.allclose(rotated['rel_l2'],np.sqrt(2),rtol=1e-6)
assert np.allclose(rotated['amplitude_rel_l2'],0)
assert np.allclose(rotated['intensity_weighted_phase_mae'],np.pi/2)
assert np.allclose(rotated['intensity_ratio'],1)
eps=np.full((160,480),2.085,dtype=np.float32); eps[75:85,:]=5
for sx in [6,474]:
    src=np.zeros_like(eps); src[75:85,sx]=1
    f=geometry_features(eps,src)
    assert np.allclose(f['path'],f['linear']), 'Straight-guide path must equal linear distance.'
manifest=json.loads((ROOT/'fresh_manifest.json').read_text(encoding='utf-8'))
for name,sha in manifest['weights'].items():
    assert hashlib.sha256((ROOT/'runs'/name/'best.pt').read_bytes()).hexdigest()==sha
verify=json.loads((ROOT/'fresh_evaluation/inference_verification.json').read_text())
assert len(verify['replay'])==15
assert max(v['prediction_replay_relative_difference'] for v in verify['replay'])<1e-5
assert verify['fresh_sha256']==hashlib.sha256((ROOT/'fresh_test.npz').read_bytes()).hexdigest()
assert manifest['eps_overlap_with_old']==[] and manifest['eps_unique_count']==37
results=json.loads((ROOT/'fresh_evaluation/results.json').read_text())
assert len(results)==15
for r in results:
    assert len(r['samples'])==37
    assert np.isclose(np.mean([v['rel_l2'] for v in r['samples']]),r['overall']['rel_l2'])
supplemental=json.loads((ROOT/'exposure_matched/frozen_weights.json').read_text())
assert len(supplemental)==6
for name,sha in supplemental.items():
    assert hashlib.sha256((ROOT/'exposure_matched/runs'/name/'best.pt').read_bytes()).hexdigest()==sha
for base in [ROOT/'runs',ROOT/'exposure_matched/runs']:
    for run in base.iterdir():
        history=json.loads((run/'history.json').read_text())
        summary=json.loads((run/'summary.json').read_text())
        assert summary['best_step']==min(history,key=lambda r:r['val_rel_l2'])['step']
write_json(ROOT/'verification.json',dict(metric_identity_tests='passed',straight_path_invariant='passed',
    frozen_weights_verified=15,prediction_replay_verified=15,fresh_geometry_count=37,
    distinct_eps_count=37,old_eps_overlap=0,per_sample_aggregation='passed',supplemental_weights_verified=6,
    validation_checkpoint_selection_verified=21))
print('Metric identities, straight-guide invariant, 21 weights/validation selections, 15 replays, and 37-sample provenance passed.')
