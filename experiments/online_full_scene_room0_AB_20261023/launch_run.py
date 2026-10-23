from pathlib import Path
import json,os,subprocess
HERE=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_full_scene_room0_AB_20261023')
OUT=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023')
for group in ['A','B']:
    assert not (OUT/('status_'+group+'.json')).exists()
    with (OUT/('driver_'+group+'.log')).open('ab',buffering=0) as log:
        proc=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(HERE/'run_full_scene.py'),group],
            cwd=HERE,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(group=group,pid=proc.pid,results=str(OUT))))
