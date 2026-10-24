"""Archive only this experiment's final small evidence and source; never push."""
from pathlib import Path
import json,shutil,subprocess,time,hashlib
R=Path('/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023')
C=R/'experiments/online_replica7_window_AB_20261023'
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
E=C/'evidence/final';E.mkdir(parents=True,exist_ok=True)
assert json.loads((O/'post_complete.json').read_text())['status']=='PASS'
assert json.loads((C/'delivery_verification.json').read_text())['status']=='PASS'
assert json.loads((C/'browser_verification.json').read_text())['status']=='PASS'
for rel in ['post_complete.json','deliverables_package.json','diagnostics.json','review/summary.json','review/evidence_manifest.json','review/frozen_kernel_audit.json','review/adapter_audit.json','review/batch_mapping_repair_complete.json','review/recovery_status.json','review/status_write_race_fix.json','review/run_plan.json']:
 p=O/rel;dest=E/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for scene in ['room1','room2','office0','office1','office2','office3','office4']:
 src=O/'review'/scene/'verification';dest=E/scene/'verification';shutil.copytree(src,dest,dirs_exist_ok=True)
rel=str(C.relative_to(R))
subprocess.run(['git','add','--',rel],cwd=R,check=True)
subprocess.run(['git','commit','--only','-m','Deliver seven-scene online A/B results, R2 diagnostics and 70-pair PLY viewer','--',rel],cwd=R,check=True,stdout=subprocess.DEVNULL)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
receipt=dict(status='PASS',commit=commit,pushed=False,unix=time.time(),scope=rel)
(O/'final_git_archive.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
