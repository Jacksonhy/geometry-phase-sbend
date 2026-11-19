"""Freeze v1 definitions/exclusions/rules; does NOT instantiate formal samples."""
import hashlib
import numpy as np
from common import ROOT,OUT,OLD,read,save,sha,now
records=read(OLD/'sbend_index.json');exclusions=[]
for r in records:
    with np.load(OLD/r['sample'],allow_pickle=False) as d:
        eps=d['eps'];fprints=[hashlib.sha256(a.tobytes()).hexdigest() for a in [eps,np.rot90(eps,2)]]
    exclusions.append({'geometry_id':r['geometry_id'],'params_grid_0025um':{k:int(round(v/.025)) for k,v in r['params'].items()},
        'eps_float32_sha256':fprints[0],'eps_180_rotation_sha256':fprints[1],
        'complete_sample_sha256':sha(OLD/r['sample'])})
save('development_exclusions.json',{'count':16,'exclude_regardless_of_input_port':True,'records':exclusions})
manifest={'version':'sbend_path_v1','frozen_utc':now(),'seed':2026,'formal_geometry_counts':{'train':64,'val':16,'test':100},
    'formal_samples_instantiated':0,'official_fdtd_samples_generated_this_turn':0,'network_training_or_inference':False,
    'old_calibration_primary_diagnostic_only':read(ROOT/'research_e3_2400/manifest.json')['calibration'],
    'formal_calibration_source':'new 64 train fields only; fixed downstream-ROI OLS; no query labels',
    'methods':['Direct','Linear','LocalTangent','Path'],'steps_per_method':2400,
    'primary_strong_comparison':'Path vs LocalTangent','supporting_comparison':'Path vs Linear; Direct',
    'path_extension':'nearest official centerline line-segment arc; global far-cladding nonsmoothness retained and disclosed',
    'implementation_risks':['Far-cladding nearest projection can change branch abruptly; do not claim globally smooth wavefront.',
                            '11 real input channel packing and shared exp prior adapter require future pretraining verification.',
                            'No performance advantage over LocalTangent established by development phase checks.'],
    'sources':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'PROTOCOL.md',OUT/'FORMAL_PROTOCOL.md',OUT/'path_geometry.py',OUT/'diagnose.py',OUT/'verify_geometry.py',
        OUT/'development_exclusions.json',ROOT/'research_e3_2400/manifest.json',ROOT/'research_v3/experiment.py',
        OLD/'sbend_development16_seed2026.npz',OLD/'experiment.py']},
    'excluded_development_ids':[r['geometry_id'] for r in records],
    'no_future_execution_authorized_by_this_manifest':True}
assert not (OUT/'frozen_protocol.json').exists(),'refuse to replace frozen version'
save('frozen_protocol.json',manifest)
print('Frozen v1 rules and 16 exclusions. No official candidate list generated.')
