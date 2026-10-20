"""Resume unchanged frozen E1 stages after a missing legacy alias check."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
C=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019');R=ROOT/'e1_repair';E=C/'e1_repair'
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env={**os.environ,'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}

def save(p,value):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)

def main():
 assert (R/'run_failure.json').exists() and (R/'pipeline_failure.json').exists()
 status=read(R/'pipeline_status.json');assert status['stage']=='strict_repair'
 completed=status['completed'];assert [s['stage'] for s in completed]==['identity','raw_repair']
 assert read(R/'tracking_queue_complete.json')['status']=='PASS'
 assert not list((R/'original_repair').rglob('*.npz'))
 freeze=read(R/'pipeline_code_freeze.json')['files']
 def verify():
  for p,h in freeze.items():assert sha(p)==h,p
 verify()
 target=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/instance_repair/frozen_e1_20261012/room0/dependencies/joint_batch_ops.py')
 alias=Path('/home/chenkejun/CVPR/experiments/room0_top30_repair_20261008/batch_6b3f32f5dc1c/joint_batch_ops.py')
 legacy=Path('/home/chenkejun/CVPR/experiments/mask_gate_no_foreign_v2_20261010/room0/joint_batch_ops.py')
 expected='c3c956071eea48ae0b13db33363eccb2d5bc05cfdf87d05f416301d975f0b19b'
 assert sha(target)==sha(legacy)==sha(E/'joint_batch_ops.py')==expected
 assert not alias.exists() and not alias.is_symlink()
 alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(target);assert sha(alias)==expected
 save(R/'legacy_path_compatibility.json',dict(status='PASS',reason='An adapted duplicate-source hash check referenced a missing legacy room0 batch path.',alias=str(alias),target=str(target),sha256=expected,source_files_unchanged=True,numerical_algorithm_unchanged=True,resume_from='strict_repair',time=time.time()))
 archive=R/'attempt1_strict_path_failure';archive.mkdir()
 for name in ['run_failure.json','pipeline_failure.json','pipeline_status.json','strict_repair.log']:
  (R/name).rename(archive/name)
 (ROOT/'remaining_failure.json').rename(archive/'remaining_failure.json')
 save(ROOT/'remaining_status.json',dict(stage='e1_controller_resumed',pid=os.getpid(),time=time.time()))
 save(R/'run_status.json',dict(status='RUNNING',stage='pipeline',resumed_from='strict_repair',time=time.time()))
 for stage in ['strict_repair','history_core','expand','E1']:
  verify();started=time.time();log=R/(stage+'.log')
  with log.open('xb') as f:
   p=subprocess.Popen([PY,'-B','-u',str(E/'run_stage.py'),stage],cwd=E,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   save(R/'pipeline_status.json',dict(status='RUNNING',stage=stage,pid=p.pid,completed=completed,log=str(log),unix=time.time()))
   assert p.wait()==0,stage+' failed; inspect '+str(log)
  completed.append(dict(stage=stage,seconds=time.time()-started));print(json.dumps(completed[-1]),flush=True)
 verify();save(R/'pipeline_complete.json',dict(status='PASS',stages=completed,GT_used=False,resumed_from='strict_repair',source_files_unchanged=True))
 save(R/'pipeline_status.json',dict(status='PASS',stage='mapping_complete',completed=completed,unix=time.time()))
 save(R/'run_status.json',dict(status='RUNNING',stage='evaluation',time=time.time()))
 with (R/'evaluation.controller.log').open('xb') as f:
  p=subprocess.run([PY,'-B','-u',str(E/'evaluate_r2.py')],cwd=E,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,'evaluation failed'
 verify();selected=read(E/'confirmed_selection.json')
 save(R/'run_complete.json',dict(status='PASS',time=time.time(),selected_seeds=len(selected['included']),excluded_seeds=len(selected['excluded']),variant='original_E1',resumed_from='strict_repair',source_files_unchanged=True))
 save(R/'run_status.json',dict(status='PASS',stage='complete',time=time.time()))
 save(ROOT/'remaining_complete.json',dict(status='PASS',time=time.time(),resumed_from='strict_repair'))
 save(ROOT/'remaining_status.json',dict(stage='complete',time=time.time()))

if __name__=='__main__':
 try:main()
 except Exception:
  save(R/'run_failure.json',dict(status='FAIL',traceback=traceback.format_exc()))
  save(ROOT/'remaining_failure.json',dict(error=traceback.format_exc(),time=time.time()))
  raise
