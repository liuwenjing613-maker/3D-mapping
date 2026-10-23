from pathlib import Path
import json,shutil,subprocess,hashlib,time
R=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
C=R/'experiments/online_replica7_window_AB_20261023';O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
E=C/'evidence/status_write_race';E.mkdir(parents=True,exist_ok=True)
for rel in ['status_write_race_fix.json','recovery_launch.json','recovery_status.json','office3/50/A/tracking/tracking_queue_failure.json']:
 p=O/rel;dest=E/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
audit=json.loads((O/'status_write_race_fix.json').read_text());assert audit['status']=='PASS'
for row in audit['files']:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['after_sha256']
rel=str(C.relative_to(R));subprocess.run(['git','add','--',rel],cwd=R,check=True)
subprocess.run(['git','commit','--only','-m','Fix concurrent status writes and resume failed tracking after active branches finish','--',rel],cwd=R,check=True,stdout=subprocess.DEVNULL)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
receipt=dict(status='PASS',commit=commit,pushed=False,unix=time.time(),scope='logging concurrency, recovery orchestration and truthful retry provenance only')
(O/'race_fix_git_archive.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
