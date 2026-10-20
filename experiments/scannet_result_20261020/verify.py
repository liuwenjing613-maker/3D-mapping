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
p=P/'external_dependencies_inventory.json'
if p.exists():
    for record in json.loads(p.read_text(encoding='utf-8'))['records']:
        assert sha(REPO/record['repository_path'])==record['sha256'],record['server_source']
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

gt_path=P/'results/gt_ply_delivery_manifest.json'
if gt_path.exists():
    gt_manifest=json.loads(gt_path.read_text(encoding='utf-8'))
    assert gt_manifest['status']=='PASS'
    assert sha(P/'tools/export_gt_ply.py')==gt_manifest['exporter_sha256']
    assert {r['scene'] for r in gt_manifest['summaries']}==set(manifest['scenes'])
    for rel,record in gt_manifest['files'].items():
        assert sha(P/'results'/rel)==record['sha256'],rel
        assert (P/'results'/rel).stat().st_size==record['bytes'],rel
    for item in gt_manifest['summaries']:
        validation=json.loads((P/'results'/item['scene']/'gt_ply/validation.json').read_text(encoding='utf-8'))
        assert validation['status']=='PASS' and validation['summary']==item
        assert validation['exporter_sha256']==gt_manifest['exporter_sha256']
    print(json.dumps(dict(gt_ply_status='PASS',gt_supplement_files=len(gt_manifest['files']),
                          summaries=gt_manifest['summaries'])))
