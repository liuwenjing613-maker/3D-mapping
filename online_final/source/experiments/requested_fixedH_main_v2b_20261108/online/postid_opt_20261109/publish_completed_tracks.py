from pathlib import Path
import json,sys,fcntl,hashlib,time,traceback,os
C=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109');R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
sys.path.insert(0,str(C));from shared_seed_tracking import signature,SeedLease,ROOT,atomic
status=ROOT/'publisher_status.json';total=0
try:
 while True:
  for n in [300,400]:
   for root in R.glob('postid*/*/'+str(n)+'/B'):
    fp=root/'tracking_sam3/frame_ids.json';sel=root/'candidates/fixedH/result.json'
    if not fp.exists() or not sel.exists():continue
    rows=json.loads(sel.read_text())['selected_records'];ready=[]
    for row in rows:
     st=root/'tracking_sam3'/row['mask_id']/'status.json'
     if st.exists() and json.loads(st.read_text()).get('status')=='PASS':ready.append(row)
    if not ready:continue
    ids=json.loads(fp.read_text())['frame_ids'];scope=signature(root/'tracking_sam3/rgb',ids,C/'sam3_seed_memory_limited.py',root.parts[-3],n)
    for row in ready:
     key=hashlib.sha256(json.dumps([scope,row['frame'],row['mask_sha256']]).encode()).hexdigest()
     if (ROOT/(key+'.json')).exists():continue
     lock=(ROOT/(key+'.lock')).open('a')
     try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:lock.close();continue
     lease=SeedLease(key,lock,root,row,scope,ids)
     try:lease.publish();total+=1
     finally:lease.close()
  controllers=[json.loads(p.read_text()) for p in R.glob('postid*/*/optimization_status.json')]
  done=len(controllers)>=12 and all(d['status']=='PASS' for d in controllers)
  atomic(status,dict(status='PASS' if done else 'RUNNING',published_new=total,updated_unix=time.time(),all_actual_online_complete=done,GT_used=False))
  if done:break
  time.sleep(30)
except BaseException:atomic(status,dict(status='FAILED',traceback=traceback.format_exc(),published_new=total));raise
