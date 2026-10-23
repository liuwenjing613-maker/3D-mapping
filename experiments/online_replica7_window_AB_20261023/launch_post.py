from pathlib import Path
import json,subprocess,time
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
p=O/'post_launch.json'
if p.exists():
 d=json.loads(p.read_text());proc=Path('/proc')/str(d['pid'])/'cmdline'
 if proc.exists() and str(C/'finish_batch.py').encode() in proc.read_bytes():print('ALREADY_RUNNING',d['pid']);raise SystemExit
assert not (O/'post_complete.json').exists()
with (O/'post.log').open('a') as log:
 proc=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(C/'finish_batch.py')],cwd=C,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=proc.pid,started_unix=time.time());p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
