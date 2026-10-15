from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
L='/home/chenkejun/CVPR/experiments/pilot10_repair_20261007'
files={**{'tracking_'+n:L+'/objectwise_repair_20261007/'+n for n in ['track_case.py','run_case.py','objectwise_ops.py','policy.json']},
 'seed_manifest_r4.json':'/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4/seed_manifest.json'}
c=connect()
try:
 with c.open_sftp() as s:
  for n,p in files.items():s.get(p,str(H/'references'/n))
 print('FETCHED',len(files))
finally:c.close()
