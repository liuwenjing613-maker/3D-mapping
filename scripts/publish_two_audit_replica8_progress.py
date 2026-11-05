import time,subprocess,sys,json
from pathlib import Path
r=Path('/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104');previous=set()
while True:
 ready={p.parent.name for p in (r/'stable_seed_filter_v1').glob('*/complete.json')}
 finished=(r/'replica8_run_status.json').exists()
 if ready!=previous or finished:
  subprocess.run([sys.executable,'-u','scripts/build_two_audit_replica8_final.py'],check=True)
  previous=ready
 if finished:break
 time.sleep(30)
