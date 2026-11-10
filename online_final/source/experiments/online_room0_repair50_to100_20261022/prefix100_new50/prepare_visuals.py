"""Prepare comparison images while numerical selection continues."""
from build_review import *

frontend=read(R/'frontend/input_manifest.json')
jobs=[j for j in frontend['jobs'] if j['kind']=='crop']
for k,job in enumerate(jobs,1):
    assert job['frame'] in FIDS
    model_assets(job)
    if k%30==0:
        write(R/'visuals_progress.json',dict(done=k,total=len(jobs)))
        print('VISUALS',k,len(jobs),flush=True)
write(R/'visuals_complete.json',dict(status='PASS',jobs=len(jobs),unique_rendered_comparisons=len(overlay_cache)))
