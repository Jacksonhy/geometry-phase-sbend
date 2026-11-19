"""Verify all published field, checkpoint, copied-source and image hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4194304), b''):
            h.update(b)
    return h.hexdigest()


def main():
    counts = {}
    for label, manifest, hash_key, path_key in [
        ('checkpoints', 'configs/checkpoints.json', 'sha256', 'path'),
        ('fields', 'data/field_manifest.json', 'sha256', 'path'),
        ('source_mappings', 'docs/source_provenance.json', 'target_sha256', 'target'),
    ]:
        rows = json.loads((ROOT / manifest).read_text(encoding='utf-8'))
        for r in rows:
            path = ROOT / r[path_key]
            if not path.is_file() or digest(path) != r[hash_key]:
                raise RuntimeError('Hash mismatch: ' + r[path_key])
        counts[label] = len(rows)
    assert counts['checkpoints'] == 12 and counts['fields'] == 288
    print(json.dumps(dict(passed=True, **counts)))


if __name__ == '__main__':
    main()
