from pathlib import Path
import os,sys,json,time,subprocess,fcntl,traceback,hashlib
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
from gpu_memory_scheduler import acquire
from shared_seed_tracking import signature,claim
root=Path(sys.argv[1]);out=root/'tracking_sam3';out.mkdir(parents=True,exist_ok=True);start=time.time();dirs=[Path('/data/chenkejun/CVPR/results/ovimap_aligned_main_onlinev2_20261101/gpu_locks'),Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/gpu_locks')];worker=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109/sam3_seed_memory_limited.py')
def write(d):q=out/'driver_status.tmp';q.write_text(json.dumps(d,indent=2)+'\n');q.replace(out/'driver_status.json')
try:
 fixed=root/'fixedH_driver_status.json'
 while True:
  d=json.loads(fixed.read_text()) if fixed.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError('fixedH failed '+str(d))
  if d['status']=='PASS':break
  write({'status':'WAITING_SCREENING','elapsed_seconds':time.time()-start});time.sleep(10)
 sel=json.loads((root/'candidates/fixedH/result.json').read_text());assert sel['status']=='PASS' and not sel['GT_used'];n=int(root.parts[-2]);cfg=json.loads((root/'mapping/config.json').read_text());source=Path(cfg['source']['scene_root']);ids=list(range(cfg['frame_selection']['start'],cfg['frame_selection']['stop_exclusive']));assert n==400 and len(ids)==1996;video=out/'rgb';video.mkdir(exist_ok=True)
 for i,fid in enumerate(ids):
  src=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/replica_sam3_tracking')/root.parts[-3]/'rgb'/f'{fid:06d}.jpg';dst=video/f'{i:06d}.jpg';assert src.exists()
  if not dst.exists():dst.symlink_to(src.resolve())
  assert dst.resolve()==src.resolve()
 (out/'frame_ids.json').write_text(json.dumps({'frame_ids':ids,'causal_checkpoint':n,'tracking_frame_scope':'all raw frames through current causal prefix','GT_used':False})+'\n')
 scope=signature(video,ids,worker,root.parts[-3],n);row_lookup={r["mask_id"]:r for r in sel["selected_records"]};seed_leases={}
 pending=[];done=[];running={};failed=[];retries={}
 for row in sel['selected_records']:
  mid=row['mask_id'];p=out/mid/'status.json'
  if p.exists() and json.loads(p.read_text()).get('status')=='PASS':
   d=json.loads(p.read_text());assert d['seed_sha256']==row['mask_sha256'] and d['frame_count']==len(ids);done.append(mid)
  else:pending.append(mid)
 while pending or running:
  for gpu,(proc,mid,log,locks) in list(running.items()):
   if proc.poll() is None:continue
   log.close();p=out/mid/'status.json';d=json.loads(p.read_text()) if p.exists() else {'status':'FAILED'}
   if proc.returncode==0 and d.get('status')=='PASS':seed_leases[mid].publish();done.append(mid)
   else:
    record={'mask':mid,'gpu':locks.data['gpu'],'exit':proc.returncode,'status':d,'unix':time.time()}
    transient='OutOfMemoryError' in d.get('traceback','') or 'CUDA out of memory' in d.get('traceback','')
    if transient and retries.get(mid,0)<2:
     retries[mid]=retries.get(mid,0)+1;record['retry']=retries[mid];pending.append(mid)
     with (out/'retry_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    else:failed.append(record)
   locks.close();seed_leases.pop(mid).close()
   del running[gpu]
  if failed and not running:raise RuntimeError('SAM3 seed tracking failed: '+str(failed[-1]))
  while pending and not failed and len(running)<4:
   selected_mid=None;seed_lease=None
   for candidate in list(pending):
    disposition,sl=claim(root,row_lookup[candidate],scope,ids)
    if disposition=='reused':done.append(candidate);pending.remove(candidate);continue
    if disposition=='claimed':selected_mid=candidate;seed_lease=sl;break
   if selected_mid is None:break
   locks=acquire({50:14000,100:14000,200:18000,300:20000,400:22000}[n],'tracking:'+str(root),blocking=False)
   if locks is None:seed_lease.close();break
   uid=locks.uid;idx=locks.data['gpu'];mid=selected_mid;pending.remove(mid);seed_leases[mid]=seed_lease
   p=out/mid;p.mkdir(exist_ok=True);log=(p/'driver.log').open('ab');env={**os.environ,'CUDA_VISIBLE_DEVICES':uid,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','LIBRARY_PATH':'/data/chenkejun/CVPR/envs/sam3_cuda_link','PYTHONPATH':'/data/chenkejun/CVPR/envs/sam3_compat:/home/chenkejun/CVPR/repos/sam3-main'}
   proc=subprocess.Popen(['/data/chenkejun/CVPR/envs/dualmap_torch27/bin/python','-B','-u','-c',"import torch,runpy,sys; torch.cuda.set_per_process_memory_fraction(20000*1048576/torch.cuda.get_device_properties(0).total_memory); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",str(worker),str(root),mid],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);locks.attach(proc.pid);running[proc.pid]=(proc,mid,log,locks)
   with (out/'gpu_events.jsonl').open('a') as f:f.write(json.dumps({'event':'start','unix':time.time(),'gpu':idx,'mask':mid,'pid':proc.pid})+'\n')
  write({'status':'RUNNING','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','input_seeds':len(sel['selected_records']),'completed':len(done),'pending':len(pending),'running':[{'gpu':lease.data['gpu'],'pid':p.pid,'mask':mid} for gpu,(p,mid,_,lease) in running.items()],'elapsed_seconds':time.time()-start,'GT_used':False,'repair_executed':False});time.sleep(10)
 write({'status':'PASS','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','completed':len(done),'input_seeds':len(sel['selected_records']),'elapsed_seconds':time.time()-start,'GT_used':False,'repair_executed':False})
except BaseException:write({'status':'FAILED','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
