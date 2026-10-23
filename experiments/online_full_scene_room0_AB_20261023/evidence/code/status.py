from pathlib import Path
import json
OUT=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
for group in ['A','B']:
    for rel in ['status_'+group+'.json']+[f'{n}/{group}/{p}' for n in [200,300,400] for p in ['mapping/mapping_progress.json','candidates/selection_progress.json','tracking/tracking_queue_status.json','tracking/progress.json','repair/progress.json','repair/commit_progress.json']]:
        p=OUT/rel
        if not p.exists():continue
        o=json.loads(p.read_text())
        print(rel,json.dumps({k:v for k,v in o.items() if k in ['status','checkpoint','stage','pid','processed_frames','seconds','completed_cases','total_cases','frames','total','traceback','retained','last_raw_frame']},ensure_ascii=False))
