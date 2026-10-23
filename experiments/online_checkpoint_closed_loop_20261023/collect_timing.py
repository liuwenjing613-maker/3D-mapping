"""Keep timing scopes explicit; include independently precomputed selection stages."""
from pathlib import Path
import json

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())

rows=[]
for mode in ['reference','checkpoint']:
    r=OUT/mode
    selection=read(r/'candidates/selection_complete.json')
    track=read(r/'tracking/complete.json')
    mapping=read(r/'mapping/mapping_timing.json')
    done=read(r/'mapping/mapping_complete.json')
    rows.append(dict(mode=mode,mapping_timing=mapping,mapping_complete_timing={k:v for k,v in done.items() if 'seconds' in k},
        candidate_production_seconds_including_precomputed_stages=selection['production_seconds'],
        candidate_successful_stage_seconds=sum(x['seconds'] for x in selection['process_timings']),
        candidate_gpu_wait_seconds=selection['gpu_wait_seconds'],
        track_orchestration_seconds=track['seconds'],tracking_inferred_cases=track['inferred_cases'],tracking_reused_cases=track['reused_cases'],
        repair_seconds=read(r/'repair_execution_timing.json')['seconds'],
        commit_seconds=read(r/'commit_execution_timing.json')['seconds'],
        frame201_mapping_timing=read(r/'frame201/mapping/mapping_timing.json')))
obj=dict(scope='Mapping timings are the earlier same-input standalone benchmark. Downstream stage measurements are this cache-reuse experiment; precomputed candidate stages are included in production_seconds. Do not sum orchestration and nested subprocess times. Concurrent control/diagnostic jobs and cache states make downstream numbers descriptive, not an isolated performance benchmark.',rows=rows)
(OUT/'timing_summary.json').write_text(json.dumps(obj,indent=2)+'\n')
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k not in ['mapping_timing','frame201_mapping_timing']},indent=2))
    print('mapping_timing',json.dumps(row['mapping_timing'])[:10000])
