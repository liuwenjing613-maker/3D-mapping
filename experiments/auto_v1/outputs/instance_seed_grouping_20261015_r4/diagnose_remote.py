from pathlib import Path
import sys,shlex
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
C='/home/chenkejun/CVPR/experiments/instance_seed_grouping_20261015_r4'
R='/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4'
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
c=connect()
try:
    _,o,e=c.exec_command('mkdir -p '+C+' '+R);assert o.channel.recv_exit_status()==0
    with c.open_sftp() as s:s.put(str(H/'diagnose_noise.py'),C+'/diagnose_noise.py')
    _,o,e=c.exec_command(PY+' '+C+'/diagnose_noise.py');out=o.read().decode();err=e.read().decode();print(out,err);assert o.channel.recv_exit_status()==0
    with c.open_sftp() as s:
        for name in s.listdir(R):
            if name=='shape_diagnostics.json' or name.startswith('shape_montage_'):s.get(R+'/'+name,str(H/name))
finally:c.close()
