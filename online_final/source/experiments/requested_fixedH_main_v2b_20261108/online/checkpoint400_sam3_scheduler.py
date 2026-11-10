from pathlib import Path
import os,sys,json,time,subprocess,fcntl,traceback,hashlib
root=Path(sys.argv[1]);out=root/'tracking_sam3';out.mkdir(parents=True,exist_ok=True);start=time.time();dirs=[Path('/data/chenkejun/CVPR/results/ovimap_aligned_main_onlinev2_20261101/gpu_locks'),Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/gpu_locks')];worker=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/first100_sam3_seed.py')
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
  src=root.parents[3]/'replica_sam3_tracking'/root.parts[-3]/'rgb'/f'{fid:06d}.jpg';dst=video/f'{i:06d}.jpg';assert src.exists()
  if not dst.exists():dst.symlink_to(src.resolve())
  assert dst.resolve()==src.resolve()
 (out/'frame_ids.json').write_text(json.dumps({'frame_ids':ids,'causal_checkpoint':n,'tracking_frame_scope':'all raw frames 0..1995 through causal prefix','GT_used':False})+'\n')
 pending=[];done=[];running={};failed=[]
 for row in sel['selected_records']:
  mid=row['mask_id'];p=out/mid/'status.json'
  if p.exists() and json.loads(p.read_text()).get('status')=='PASS':
   d=json.loads(p.read_text());assert d['seed_sha256']==row['mask_sha256'] and d['frame_count']==len(ids);done.append(mid)
  else:pending.append(mid)
 while pending or running:
  for gpu,(proc,mid,log,locks) in list(running.items()):
   if proc.poll() is None:continue
   log.close();p=out/mid/'status.json';d=json.loads(p.read_text()) if p.exists() else {'status':'FAILED'}
   if proc.returncode==0 and d.get('status')=='PASS':done.append(mid)
   else:failed.append({'mask':mid,'gpu':gpu,'exit':proc.returncode,'status':d})
   for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
   del running[gpu]
  if failed:raise RuntimeError('SAM3 seed tracking failed: '+str(failed[-1]))
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);busy={line.split(',')[0].strip() for line in apps.splitlines() if ',' in line}
  for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines():
   if not pending or len(running)>=4:break
   idx,uid,free,util=[s.strip() for s in row.split(',')];idx=int(idx)
   if idx in running or uid in busy or int(free)<43000 or int(util)>15:continue
   locks=[]
   try:
    for d in dirs:
     d.mkdir(parents=True,exist_ok=True);f=(d/(uid+'.lock')).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
   except BlockingIOError:
    for f in locks:fcntl.flock(f,fcntl.LOCK_UN);f.close()
    continue
   mid=pending.pop(0);p=out/mid;p.mkdir(exist_ok=True);log=(p/'driver.log').open('ab');env={**os.environ,'CUDA_VISIBLE_DEVICES':uid,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','LIBRARY_PATH':'/data/chenkejun/CVPR/envs/sam3_cuda_link','PYTHONPATH':'/data/chenkejun/CVPR/envs/sam3_compat:/home/chenkejun/CVPR/repos/sam3-main'}
   proc=subprocess.Popen(['/data/chenkejun/CVPR/envs/dualmap_torch27/bin/python','-B','-u',str(worker),str(root),mid],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[idx]=(proc,mid,log,locks)
   with (out/'gpu_events.jsonl').open('a') as f:f.write(json.dumps({'event':'start','unix':time.time(),'gpu':idx,'mask':mid,'pid':proc.pid})+'\n')
  write({'status':'RUNNING','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','input_seeds':len(sel['selected_records']),'completed':len(done),'pending':len(pending),'running':[{'gpu':gpu,'pid':p.pid,'mask':mid} for gpu,(p,mid,_,_) in running.items()],'elapsed_seconds':time.time()-start,'GT_used':False,'repair_executed':False});time.sleep(10)
 write({'status':'PASS','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','completed':len(done),'input_seeds':len(sel['selected_records']),'elapsed_seconds':time.time()-start,'GT_used':False,'repair_executed':False})
except BaseException:write({'status':'FAILED','stage':'SAM3_TRACKING_ONLY_NOT_REPAIR','traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start});raise
