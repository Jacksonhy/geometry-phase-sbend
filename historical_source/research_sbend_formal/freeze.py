from common import *
def main():
    assert read(OUT/'preflight.json')['passed']
    assert read(OUT/'runtime_checks.json')['passed']
    assert not (OUT/'code_freeze.json').exists()
    sources=list(OUT.glob('*.py'))+[OUT/'PROTOCOL.md',OUT/'config.json']
    sources.extend([V1/'path_geometry.py',V1/'common.py',V1/'FORMAL_PROTOCOL.md',V2/'path_v2.py',V2/'FORMAL_PROTOCOL_ADDENDUM.md',
        ROOT/'research_sbend_path_v3/FORMAL_DECISION.md',ROOT/'research_official_audit/validate_sparams.py',
        ROOT/'research_v4/official_architecture.py',ROOT/'research_v4/official_bridge.py'])
    sources.extend(p for p in (SRC/'FDTD').rglob('*') if p.is_file() and '__pycache__' not in str(p))
    sources.extend(ROOT/'references/NeurOLight-source/NeurOLight-main'/p for p in ['core/models/pde_base.py','core/models/neurolight_cnn.py','core/models/layers/fno_conv2d.py','core/models/layers/neurolight_conv2d.py'])
    write('code_freeze.json',{'time_utc':now(),'seed':SEED,'sources':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources))},
        'config':read(OUT/'config.json'),'preflight_sha256':sha(OUT/'preflight.json'),
        'runtime_checks_sha256':sha(OUT/'runtime_checks.json'),'fdtd_environment_sha256':sha(OUT/'fdtd_environment.json'),'v3_included':False,
        'before_formal_geometries':not (OUT/'geometry_manifest.json').exists()})
    print('EXECUTABLE CODE FROZEN',flush=True)
if __name__=='__main__':main()
