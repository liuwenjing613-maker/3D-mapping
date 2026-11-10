"""Exact-seed tracking reuse across this experiment's threshold arms."""
from pathlib import Path
import hashlib,json,os,fcntl,time
ROOT=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109/shared_seed_cache')
ROOT.mkdir(exist_ok=True)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def atomic(p,d):
 q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def signature(video,ids,worker,scene,n):
 lock=(ROOT/'source_signature.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  p=ROOT/(scene+'_'+str(n)+'_source.json')
  paths=[(video/f'{i:06d}.jpg').resolve() for i in range(len(ids))]
  paths+=[worker.resolve(),Path('/data/chenkejun/CVPR/models/sam3.pt')]
  stats=[(str(x),x.stat().st_size,x.stat().st_mtime_ns) for x in paths]
  old=json.loads(p.read_text()) if p.exists() else {}
  if old.get('stats')!=[list(x) for x in stats]:
   hashes=[digest(x) for x in paths]
   value=hashlib.sha256(json.dumps([scene,n,ids,hashes]).encode()).hexdigest()
   atomic(p,dict(stats=stats,signature=value,source_sha256=dict(zip(map(str,paths),hashes))))
  else:value=old['signature']
  return value
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
class SeedLease:
 def __init__(self,key,lock,root,row,scope,ids):self.key=key;self.lock=lock;self.root=root;self.row=row;self.scope=scope;self.ids=ids
 def publish(self):
  p=self.root/'tracking_sam3'/self.row['mask_id'];d=json.loads((p/'status.json').read_text())
  assert d['status']=='PASS' and d['seed_sha256']==self.row['mask_sha256'] and d['seed_frame']==self.row['frame'] and d['frame_count']==len(self.ids)
  assert {x.name for x in (p/'masks').glob('*.png')}=={f'{fid:06d}.png' for fid in self.ids}
  atomic(ROOT/(self.key+'.json'),dict(source=str(p.resolve()),scope_signature=self.scope,seed_sha256=d['seed_sha256'],seed_frame=d['seed_frame'],frame_count=d['frame_count'],GT_used=False))
 def close(self):fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
def claim(root,row,scope,ids):
 # A failed scheduler can leave a healthy detached child; never duplicate its output writes.
 dst=root/'tracking_sam3'/row['mask_id'];st=dst/'status.json'
 if st.exists() and json.loads(st.read_text()).get('status')=='RUNNING':
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:
    args=(proc/'cmdline').read_bytes().split(bytes([0]))
    if str(root).encode() in args and row['mask_id'].encode() in args and any(b'sam3_seed_memory_limited.py' in a for a in args):return 'busy',None
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 key=hashlib.sha256(json.dumps([scope,row['frame'],row['mask_sha256']]).encode()).hexdigest()
 lock=(ROOT/(key+'.lock')).open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();return 'busy',None
 lease=SeedLease(key,lock,root,row,scope,ids);index=ROOT/(key+'.json')
 # A detached child may have finished after this scheduler built its pending list.
 if st.exists() and json.loads(st.read_text()).get('status')=='PASS':
  lease.publish();lease.close();return 'reused',None
 if index.exists():
  d=json.loads(index.read_text());p=Path(d['source']);st=json.loads((p/'status.json').read_text())
  assert d['scope_signature']==scope and st['status']=='PASS' and st['seed_sha256']==row['mask_sha256'] and st['seed_frame']==row['frame'] and st['frame_count']==len(ids)
  dst=root/'tracking_sam3'/row['mask_id']
  if dst.exists() and dst.resolve()!=p.resolve():
   prior=json.loads((dst/'status.json').read_text()) if (dst/'status.json').exists() else {}
   assert prior.get('status')!='PASS'
   dst.rename(dst.with_name(dst.name+'.incomplete_'+str(time.time_ns())))
  if not dst.exists():dst.symlink_to(p.resolve(),target_is_directory=True)
  with (root/'tracking_sam3/exact_seed_reuse.jsonl').open('a') as f:f.write(json.dumps(dict(mask_id=row['mask_id'],source=str(p),scope_signature=scope,seed_sha256=row['mask_sha256'],seed_frame=row['frame'],frame_count=len(ids)))+'\n')
  lease.close();return 'reused',None
 return 'claimed',lease
