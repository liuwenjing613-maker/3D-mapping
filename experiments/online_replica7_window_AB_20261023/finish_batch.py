"""Wait for complete branch commits, then audit, main R2 and lossless exports."""
from pathlib import Path
import os,sys,json,time,subprocess,traceback,fcntl
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023')
SCENES=['room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def run(script,args):
 write(O/'post_status.json',dict(status='RUNNING',stage=script,args=args,pid=os.getpid(),updated_unix=time.time()))
 subprocess.run([sys.executable,'-B','-u',str(C/script),*map(str,args)],cwd=C,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'},check=True)
def main():
 lock=(O/'post.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  complete=[];waiting=[]
  for scene in SCENES:
   for g in ['A','B']:
    p=O/scene/f'post_{g}_complete.json'
    if p.exists():assert read(p)['status']=='PASS';complete.append([scene,g]);continue
    status=O/scene/f'status_{g}.json'
    if not status.exists() or read(status)['status']!='COMPLETE':waiting.append([scene,g]);continue
    run('audit_branch.py',[scene,g])
    e=O/scene/'evaluation_r2'/g/'evaluation_complete.json'
    if not e.exists():run('evaluate_branch.py',[scene,g])
    assert read(e)['status']=='PASS'
    for n in [50,100,200,300,400]:
     for stage in ['before','after']:
      m=O/'deliverables'/scene/f'{scene}_{g}_{n}_{stage}_export.json'
      if not m.exists():run('export_checkpoint.py',[scene,g,n,stage])
      assert read(m)['all_points_and_instance_ids_roundtrip_exact']
    write(p,dict(status='PASS',scene=scene,branch=g,checkpoints=[50,100,200,300,400],R2_complete=True,PLY_exports=10,finished_unix=time.time()))
    complete.append([scene,g])
  if len(complete)==14:
   run('collect_summary.py',[])
   write(O/'post_complete.json',dict(status='PASS',branches=complete,checkpoints=70,PLY_exports=140,summary=str(O/'review/summary.json'),finished_unix=time.time()))
   write(O/'post_status.json',dict(status='PASS',finished_unix=time.time()));return
  write(O/'post_status.json',dict(status='WAITING_BRANCH_COMPLETION',complete=complete,waiting=waiting,pid=os.getpid(),updated_unix=time.time()))
  time.sleep(300)
if __name__=='__main__':
 try:main()
 except Exception:write(O/'post_status.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise
