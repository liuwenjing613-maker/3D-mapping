from pathlib import Path
import json
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
old=read(H.parent/'e1_roi_top10_20261014/input_manifest.json');new=read(H/'input_manifest.json')
before={j['key']:j for j in old['jobs']};after={j['key']:j for j in new['jobs']}
assert before.keys()<=after.keys()
fields=['RGB_sha256','baseline_sha256','crop_xyxy','frame','strategy','shape']
for key,o in before.items():
 for f in fields:assert o[f]==after[key][f],(key,f)
for g in ['CONFLICT','U','T']:
 ranks=sorted(r['rank'] for c in new['cases'] for r in c['rankings'] if r['group']==g)
 assert ranks==list(range(1,21))
report=dict(status='PASS',rank_entries=60,unique_ROIs=len(new['cases']),view_entries=sum(len(c['views']) for c in new['cases']),crop_count=len(after)-1,added_crops=len(after)-len(before),old_jobs_exactly_preserved=len(before),fields_verified=fields,source_surface_unchanged=old['source_surface_sha256']==new['source_surface_sha256'],reused_prediction_occurrences=len(new['cache_reuse']))
assert report['source_surface_unchanged']
(H/'expansion_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
