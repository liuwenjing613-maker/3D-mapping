from pathlib import Path
import os,subprocess,json
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_full_scene_room0_AB_20261023/window_only_repair400')
O=Path('/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023/window_only_repair400')
for branch in ['A','B']:
 root=O/'400'/branch;receipt=root/'launch.json'
 if receipt.exists():
  print('EXISTS',receipt.read_text());continue
 env={**os.environ,'CVPR_CHECKPOINT':'400','CVPR_TRACK_GROUP':branch,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
 env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with (root/'driver.log').open('ab') as log:
  p=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(C/'run_window_repair400.py')],cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(branch=branch,pid=p.pid,code=str(C),root=str(root));receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
