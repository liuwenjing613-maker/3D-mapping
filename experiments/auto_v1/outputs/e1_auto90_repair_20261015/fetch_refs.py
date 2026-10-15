from pathlib import Path
import sys,json
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'replica_gt_scope_20261008'))
from ssh_task import connect
M='/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/instance_repair'
B='/home/chenkejun/CVPR/experiments/room2_strict_manual_ab_20261010/batch_20fd547a5808'
V='/home/chenkejun/CVPR/experiments/mask_gate_no_foreign_v2_20261010/room2'
F=M+'/frozen_e1_20261012/room2'
files={**{'frozen_'+n:F+'/'+n for n in ['expand_room2.py','adaptive_core.py','observation_ops.py','run_observation_repair.py','original_policy.json','module_map.json']},
       **{'batch_'+n:B+'/'+n for n in ['prepare_server.py','tracking_queue.py','run_batch.py','pipeline_ab.py','prepare_local.py']},
       **{'gate_'+n:V+'/'+n for n in ['gate_policy.json','gate_v1.py','run_batch.py','run_strict_repair.py','joint_batch_ops.py']},
       **{'dep_'+n:F+'/dependencies/'+n for n in ['run_repairs.py','fixed_surface_repair.py','run_history_core.py','history_core_ops.py','repair_association.py']},
       'frozen_cli.py':M+'/cli.py','source_lock.json':M+'/frozen_e1_20261012/source_lock.json'}
c=connect()
try:
    with c.open_sftp() as s:
        out=H/'references';out.mkdir(exist_ok=True)
        for name,p in files.items():s.get(p,str(out/name))
    (out/'paths.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print('DOWNLOADED_REFERENCES',len(files))
finally:c.close()
