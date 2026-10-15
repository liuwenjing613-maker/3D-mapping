from pathlib import Path
import sys,shlex,json
h=Path(__file__).resolve().parent;sys.path.insert(0,str(h.parent/'replica_gt_scope_20261008'));from ssh_task import connect
c=connect()
try:
 cmd="from pathlib import Path; p=Path('/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010/common.py'); t=p.read_text(); a=t.index('def project_visible'); b=t.find('\\ndef ',a+1); print(t[a:b if b!=-1 else None]); import numpy as np; z=np.load('/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015/frame000735_visible.npz'); print({k:(z[k].shape,str(z[k].dtype)) for k in z.files}); p=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/final/instance_surface.npz'); z=np.load(p); print('surface keys',z.files);print('surface_shape',z['xyz_m'].shape)"
 _,o,e=c.exec_command('/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python -c '+shlex.quote(cmd));print(o.read().decode(),e.read().decode());assert o.channel.recv_exit_status()==0
finally:c.close()
