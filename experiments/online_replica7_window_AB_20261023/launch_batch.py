from pathlib import Path
import os,json,subprocess,time
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
p=O/'launch.json'
if p.exists():
 old=json.loads(p.read_text());proc=Path('/proc')/str(old['pid'])/'cmdline'
 if proc.exists() and str(C/'run_batch.py').encode() in proc.read_bytes():
  print('ALREADY_RUNNING',old['pid']);raise SystemExit
assert not (O/'batch_mapping_repair_complete.json').exists()
with (O/'supervisor.log').open('a') as log:
 q=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(C/'run_batch.py')],cwd=C,
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=q.pid,started_unix=time.time(),code=str(C),results=str(O));p.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
