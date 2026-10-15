from pathlib import Path
import json,statistics,time
R=Path('/data/chenkejun/CVPR/results/e1_auto90_repair_20261015')
cases=json.loads((R/'seed_diagnostics.json').read_text())['cases'];rows=[]
adopt=json.loads((R/'queue_v2_adoption.json').read_text());changed=adopt['old_queue_stopped_at']
launches=json.loads((R/'tracking_launch_v2.json').read_text())['tasks'];after={x['case_uid'] for x in launches}
for case in cases:
 p=R/'cases'/case['case_uid']/'tracking/complete.json'
 if not p.exists():continue
 d=json.loads(p.read_text());rows.append({'case':case['case_uid'],'objects':len(case['objects']),'seconds':d['elapsed_seconds'],'seed_frame':d['seed_frame'],'parallel_v2':case['case_uid'] in after})
def stats(q):
 v=sorted(x['seconds'] for x in q)
 return {'tasks':len(v),'min':min(v) if v else None,'median':statistics.median(v) if v else None,'max':max(v) if v else None,'mean':statistics.mean(v) if v else None,'mean_ms_per_raw_frame_including_init_IO':statistics.mean(v)*1000/2000 if v else None}
out={'all':stats(rows),'single':stats([x for x in rows if x['objects']==1]),'single_two_jobs_per_gpu':stats([x for x in rows if x['objects']==1 and x['parallel_v2']]),'multi':stats([x for x in rows if x['objects']>1]),'by_object_count':{str(n):stats([x for x in rows if x['objects']==n]) for n in sorted(set(x['objects'] for x in rows))},'recent':rows[-6:]}
print(json.dumps(out,ensure_ascii=False))
