from pathlib import Path
import sys,shlex
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
C='/home/chenkejun/CVPR/experiments/e1_auto90_repair_20261015'
c=connect()
try:
 with c.open_sftp() as s:
  for arg in sys.argv[1:]:
   name,remote=arg.split(':',1) if ':' in arg else (arg,arg)
   s.put(str(H/name),C+'/'+remote)
   print('uploaded',remote)
finally:c.close()
