"""Memory-based GPU admission for this experiment, including startup reservations."""
from pathlib import Path
import os,json,time,subprocess,fcntl,contextlib,uuid
ROOT=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109/gpu_memory_scheduler')
ROOT.mkdir(parents=True,exist_ok=True)
LEGACY=[Path('/data/chenkejun/CVPR/results/ovimap_aligned_main_onlinev2_20261101/gpu_locks'),Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/gpu_locks')]
def alive(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().split()[2]!='Z'
 except (FileNotFoundError,ProcessLookupError):return False
@contextlib.contextmanager
def mutex():
 f=(ROOT/'admission.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX)
 try:yield
 finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()
def save(path,data):
 q=path.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2));q.replace(path)
def snapshot():
 g=[]
 for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
  idx,uid,free,total,util=[x.strip() for x in line.split(',')];g.append(dict(index=int(idx),uuid=uid,free=int(free),total=int(total),util=int(util)))
 apps={}
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines():
  cols=[x.strip() for x in line.split(',')]
  if len(cols)==3 and cols[1].isdigit() and cols[2].isdigit():apps[(cols[0],int(cols[1]))]=int(cols[2])
 return g,apps
class Lease:
 def __init__(self,path,data,locks):self.path=path;self.data=data;self.locks=locks
 @property
 def uid(self):return self.data['uuid']
 @property
 def cap_mib(self):return self.data['budget_mib']-2000
 def attach(self,pid):
  with mutex():self.data.update(child_pid=int(pid));save(self.path,self.data)
 def close(self):
  with mutex():self.path.unlink(missing_ok=True)
  for f in self.locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
def acquire(budget_mib,tag,status_root=None,blocking=True):
 started=time.monotonic()
 while True:
  with mutex():
   gpus,apps=snapshot();leases=[]
   for p in ROOT.glob('lease_*.json'):
    d=json.loads(p.read_text())
    if not alive(d['owner_pid']) and not alive(d.get('child_pid',0)):p.unlink();continue
    leases.append(d)
   allowed=set(map(int,os.environ.get('CVPR_GPU_POOL','0,1,2,3,4,5,6,7').split(',')))
   for g in sorted(gpus,key=lambda x:(-x['free'],x['util'],x['index'])):
    if g['index'] not in allowed:continue
    unmaterialized=sum(max(0,d['budget_mib']-apps.get((g['uuid'],d.get('child_pid',0)),0)) for d in leases if d['uuid']==g['uuid'] and not (d.get('reservation_only') and d['owner_pid']==os.getpid()))
    owned={d.get('child_pid') for d in leases if d['uuid']==g['uuid']}
    external_startup=0
    for (app_uid,pid),used in apps.items():
     if app_uid!=g['uuid'] or pid in owned:continue
     try:cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
     except FileNotFoundError:continue
     budget=40000 if 'infer_cropformer.py' in cmd else 14000 if '/frontend/infer_sam.py' in cmd else 22000 if ('sam3_seed' in cmd or 'sam3_seed_memory_limited' in cmd) else 0
     external_startup+=max(0,budget-used)
    effective=g['free']-unmaterialized-external_startup
    if effective<budget_mib:continue
    locks=[];locked_elsewhere=False
    for base in LEGACY:
     lockpath=base/(g['uuid']+'.lock')
     if not lockpath.exists() and 'ovimap_aligned_main_onlinev2_20261101' in str(base):continue
     f=lockpath.open('r' if lockpath.exists() else 'a')
     try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
     except BlockingIOError:f.close();locked_elsewhere=True
    # A legacy launcher may have acquired a lock but not allocated its model yet.
    if locked_elsewhere and g['free']>43000 and not any(d['uuid']==g['uuid'] for d in leases):
     for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
     continue
    data=dict(owner_pid=os.getpid(),child_pid=None,uuid=g['uuid'],gpu=g['index'],budget_mib=budget_mib,tag=tag,started_unix=time.time(),free_mib=g['free'],effective_free_mib=effective,wait_seconds=time.monotonic()-started)
    for prior in ROOT.glob('lease_*.json'):
     old=json.loads(prior.read_text())
     if old.get('reservation_only') and old['owner_pid']==os.getpid():prior.unlink()
    path=ROOT/('lease_'+uuid.uuid4().hex+'.json');save(path,data)
    if status_root:save(Path(status_root)/'gpu_wait.json',dict(status='ACQUIRED_MEMORY_BASED',**data))
    return Lease(path,data,locks)
   # Let a waiting large model drain one GPU, instead of starving behind endless small jobs.
   # The reservation is visible to earlier scheduler versions through the same lease registry.
   if budget_mib>=36000 and blocking and not any(d.get('reservation_only') for d in leases):
    possible=[]
    for g in gpus:
     if g['index'] not in allowed or g['index'] in (1,4):continue
     persistent=0
     for (uid,pid),used in apps.items():
      if uid!=g['uuid']:continue
      try:cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode(errors='replace')
      except FileNotFoundError:continue
      if not any(x in cmd for x in ['sam3_seed','first100_sam3','/frontend/infer_','/semantic_route/','background_']):persistent+=used
     if g['total']-persistent>=budget_mib:possible.append(g)
    if possible:
     g=max(possible,key=lambda x:(x['free'],-x['util']))
     data=dict(owner_pid=os.getpid(),child_pid=None,uuid=g['uuid'],gpu=g['index'],budget_mib=budget_mib,tag=tag,started_unix=time.time(),reservation_only=True)
     save(ROOT/('lease_reservation_'+str(os.getpid())+'.json'),data)
  if not blocking:return None
  if status_root:save(Path(status_root)/'gpu_wait.json',dict(status='WAITING_MEMORY',budget_mib=budget_mib,seconds=time.monotonic()-started,requirement='enough remaining memory with startup reservations; no empty-card or utilization condition'))
  time.sleep(5)
