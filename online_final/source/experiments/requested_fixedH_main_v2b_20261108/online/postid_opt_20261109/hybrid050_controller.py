from pathlib import Path
import os,sys,json,time,subprocess,traceback,hashlib,fcntl
C=Path('/home/chenkejun/CVPR/experiments/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109');R=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/online/postid_opt_20261109')
A='postid_d050_n500';H='postid_early050_n500_late050_n350';scene=sys.argv[1];source=R/A/scene;dest=R/H/scene;root=dest/'400/B';branch=C/H/'branches'/scene;code=C/H
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python';base=C.parent;start=time.time();status=dest/'optimization_status.json'
lock=(dest/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def put(p,d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def state(s,**kw):put(status,dict(status=s,variant=H,scene=scene,schedule=[100,300,400],elapsed_seconds=time.time()-start,GT_used_for_repair=False,early_commits_reused_from=str(source),**kw))
def wait(p):
 while True:
  d=read(p) if p.exists() else {'status':'WAITING'}
  if d['status']=='FAILED':raise RuntimeError(str(p)+' '+str(d))
  if d['status']=='PASS':return
  state('WAITING',stage='identical_source_prefix_and400_inputs',waiting_for=str(p));time.sleep(15)
def run(stage,script,args=(),marker=None):
 if marker is not None and marker.exists() and read(marker).get('status')=='PASS':return
 state('RUNNING',stage=stage,checkpoint=400)
 env=dict(os.environ,CVPR_CHECKPOINT='400',CVPR_TRACK_GROUP='B',CVPR_SCENE=scene,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None);env.pop('CVPR_ONLINE_GUARD',None)
 with (root/(stage+'.log')).open('ab') as f:p=subprocess.run([PY,'-B','-u',str(script),*map(str,args)],cwd=branch,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(stage+' failed: '+str(root/(stage+'.log')))
try:
 wait(source/'400/B/fixedH_driver_status.json')
 proof=[]
 for n in [100,300]:
  p=source/str(n)/'B/repair/commit_complete.json';d=read(p);assert d['status']=='PASS' and d['processed_frames']==n
  f=read(source/str(n)/'B/candidates/fixedH/result.json')['threshold_tuning'];assert f['inherit_dice_strict']==.5 and f['min_new_support']==500 and f['id_assignment_after_sameview']
  for k in ['checkpoint','source_pixel_support','journal_manifest','geometry','final_map']:assert sha(d[k])==d[k+'_sha256']
  proof.append(dict(checkpoint=n,actual_commit=str(p),commit_sha256=sha(p),thresholds=f))
 prior=read(source/'400/B/mapping/prior_commit.json');assert prior['prior_commit_sha256']==proof[-1]['commit_sha256'] and prior['processed_frames']==300
 assert read(source/'400/B/mapping_driver_status.json')['status']=='PASS'
 if not (root/'mapping').exists():(root/'mapping').symlink_to((source/'400/B/mapping').resolve(),target_is_directory=True)
 candidates=root/'candidates';candidates.mkdir(exist_ok=True)
 for p in (source/'400/B/candidates').iterdir():
  if p.name=='fixedH':continue
  dst=candidates/p.name
  if not dst.exists():dst.symlink_to(p.resolve(),target_is_directory=p.is_dir())
 put(root/'identical_prefix_proof.json',dict(status='PASS',commits=proof,mapping400_prior_commit_sha256=prior['prior_commit_sha256'],mapping400_before_any400_repair=True,source_geometry_and_candidates=str(source/'400/B'),GT_used=False))
 run('fixedH400',code/'fixedh_geometry_400.py',[scene],root/'candidates/fixedH/status.json')
 d=read(root/'candidates/fixedH/result.json');assert d['status']=='PASS' and not d['GT_used'] and all(1500<=r['frame']<=1995 for r in d['selected_records'])
 assert d['threshold_tuning']['inherit_dice_strict']==.5 and d['threshold_tuning']['min_new_support']==350
 put(root/'fixedH_driver_status.json',dict(status='PASS',stage='SCREENING_ONLY_NOT_REPAIR',GT_used=False))
 # If the lower cutoff changes no selected seed or identity, the complete computation is identical.
 orig=read(source/'400/B/candidates/fixedH/result.json')
 def numerical(rows):
  keys=['mask_sha256','frame','id_action','target_id','relation_group','support']
  return {r['mask_id']:{k:r.get(k) for k in keys} for r in rows}
 same=numerical(d['selected_records'])==numerical(orig['selected_records'])
 put(root/'last_threshold_equivalence.json',dict(identical_selected_seeds_and_IDs=same,source=str(source/'400/B'),GT_used=False))
 if same:
  wait(source/'optimization_status.json')
  for p in (source/'400/B').iterdir():
   if p.name in ['candidates','mapping']:continue
   dst=root/p.name
   if not dst.exists():dst.symlink_to(p.resolve(),target_is_directory=p.is_dir())
  assert read(root/'evaluation_current_r2/status.json')['status']=='PASS'
  state('PASS',stage='VERIFIED_IDENTICAL_ALL_CHECKPOINTS_REUSES_COMPLETE_ACTUAL_ONLINE_R2',all_checkpoint_actual_results_reused=True)
  raise SystemExit(0)

 run('tracking400',code/'sam3_scheduler_400.py',[root],root/'tracking_sam3/driver_status.json')
 run('gate400',base/'first100_gate_driver.py',[root],root/'gate_driver_status.json')
 run('repair_commit400',branch/'repair_supervisor.py',[root,branch],root/'actual_repair_driver_status.json')
 assert read(root/'repair/commit_complete.json')['processed_frames']==400
 run('postprocess400',base/'replica_online_postprocess.py',[root,branch],root/'postprocess_final/status.json')
 run('evaluate400',branch/'evaluate.py',[scene],root/'evaluation_current_r2/status.json')
 assert read(root/'evaluation_current_r2/status.json')['status']=='PASS'
 state('PASS',stage='REAL_ONLINE_100_300_400_SHARED_IDENTICAL_EARLY_COMMITS_FINALB_DBSCAN_MAIN_R2',thresholds={'100,300':'Dice>.50 new>=500','400':'Dice>.50 new>=350'})
except Exception:state('FAILED',traceback=traceback.format_exc());raise
