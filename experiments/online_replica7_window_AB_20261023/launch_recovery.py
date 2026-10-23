from pathlib import Path
import json,subprocess,time
C=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023')
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
p=O/'recovery_launch.json'
if p.exists():
 d=json.loads(p.read_text());proc=Path('/proc')/str(d['pid'])/'cmdline'
 if proc.exists() and str(C/'resume_after_drain.py').encode() in proc.read_bytes().split(b'\0'):print('ALREADY_RUNNING',d['pid']);raise SystemExit
if (O/'recovery_status.json').exists():assert json.loads((O/'recovery_status.json').read_text())['status']!='RESUMED'
with (O/'recovery.log').open('a') as log:
 q=subprocess.Popen(['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','-B','-u',str(C/'resume_after_drain.py')],cwd=C,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=q.pid,started_unix=time.time());p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
