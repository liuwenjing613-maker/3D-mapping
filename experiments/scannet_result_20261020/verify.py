"""Read-only integrity check; no server access or model dependency required."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
REPO=P.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((P/'archive_manifest.json').read_text(encoding='utf-8'))
for rel,record in manifest['files'].items():
    assert sha(P/rel)==record['sha256'],rel
    assert (P/rel).stat().st_size==record['bytes'],rel
inventory=json.loads((P/'source_inventory.json').read_text(encoding='utf-8'))
for record in inventory['records']:
    assert sha(REPO/record['repository_path'])==record['sha256'],record['runtime_relative_path']
summary=json.loads((P/'results/comparison_all.json').read_text(encoding='utf-8'))
assert summary['completed_scenes']==manifest['scenes']
assert summary['primary_baseline']=='strict_one_vote_no_repair'
for s in manifest['scenes']:
    rows=[r for r in summary['rows'] if r['scene']==s and r['stage']=='final']
    assert {r['method'] for r in rows}=={'baseline','E1','swap_partial'}
    assert {r['GT'] for r in rows}=={27 if s=='scene0011_00' else 34}
    for row in rows:
        expected=2*row['TP']/(2*row['TP']+row['FP']+row['FN'])
        assert abs(row['F1']-expected)<1e-12
print(json.dumps(dict(status='PASS',scenes=manifest['scenes'],files=len(manifest['files']),
                      source_files=len(inventory['records']),inherited_sources=inventory['unchanged'])))
