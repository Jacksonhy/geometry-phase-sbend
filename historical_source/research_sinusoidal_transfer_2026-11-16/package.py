"""Verified compact evidence package; raw arrays remain in the experiment folder."""
from pathlib import Path
import json,hashlib,zipfile,time
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
include=[]
for p in R.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(R)
 if rel.parts[0] in ['inputs','predictions','data','fine','logs','__pycache__']:continue
 if p.suffix in ['.npz','.interrupted','.zip']:continue
 if p.name in ['PACKAGE_MANIFEST.json','package_verification.json']:continue
 include.append(p)
# Per-simulation scalar records preserve source/port/QC and timings without raw arrays.
include+=list((R/'data').glob('*/result.json'))+list((R/'fine').glob('*/res*/result.json'))
include=sorted(set(include));manifest={p.relative_to(R).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in include}
(R/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
output=R/'sinusoidal_transfer_evidence_2026-11-16.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in include+[R/'PACKAGE_MANIFEST.json']:z.write(p,p.relative_to(R).as_posix())
with zipfile.ZipFile(output) as z:
 assert z.testzip() is None
 for p in include:assert hashlib.sha256(z.read(p.relative_to(R).as_posix())).hexdigest()==manifest[p.relative_to(R).as_posix()]['sha256']
(R/'package_verification.json').write_text(json.dumps(dict(passed=True,files=len(include)+1,bytes=output.stat().st_size,sha256=sha(output),raw_arrays_and_weights_included=False),indent=2)+'\n',encoding='utf-8')
print('PACKAGED',len(include)+1,output.stat().st_size)
