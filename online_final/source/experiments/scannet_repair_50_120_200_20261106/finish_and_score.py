from pathlib import Path
import sys,json,time,traceback,subprocess,hashlib,os
import numpy as np
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
OLD=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104')
OLDC=Path('/home/chenkejun/CVPR/experiments/scannet_ovimap_strict_dbscan_20261104')
sys.path.insert(0,str(OLDC/'policies/main'))
from scene_completion.pipeline import manifest_from_commit,finalize
variant,scene=sys.argv[1:];method,group=('main','A') if variant=='main' else ('online','B')
root=O/method/scene/'200'/group;out=O/'published'/variant/scene;out.mkdir(parents=True,exist_ok=True);status=out/'finish_status.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def state(s,**kw):
 q=status.with_suffix('.tmp');q.write_text(json.dumps(dict(status=s,variant=variant,scene=scene,unix=time.time(),**kw),indent=2));q.replace(status)
try:
 marker=root/'repair_handoff_status.json'
 while True:
  if marker.exists():
   d=read(marker)
   if d['status']=='FAILED':raise RuntimeError('Final repair failed: '+str(marker))
   if d['status']=='PASS':break
  supervisor=O/(scene+'_online_continuation_status.json')
  if variant=='online_B' and supervisor.exists() and read(supervisor)['status']=='FAILED':raise RuntimeError('Online continuation failed')
  state('WAITING_REAL_FINAL_REPAIR');time.sleep(15)
 receipt=read(root/'repair/commit_complete.json');assert receipt['status']=='PASS' and receipt['processed_frames']==200
 manifest=out/'scene_complete.json'
 if not manifest.exists():manifest.write_text(json.dumps(manifest_from_commit(root,scene=scene,workflow='main' if method=='main' else 'online-v2'),indent=2))
 state('RUNNING',stage='final_B');final=finalize(manifest,out/'final_B');assert final['status']=='PASS' and not final['online_state_updated'] and final['C_enabled'] is False
 prediction=Path(final['final_surface']);base=OLD/'published/no_repair'/scene/'final_B/instance_surface.npz'
 with np.load(prediction,allow_pickle=False) as a,np.load(base,allow_pickle=False) as b:
  np.testing.assert_array_equal(a['xyz_m'],b['xyz_m']);np.testing.assert_array_equal(a['rgb'],b['rgb'])
 gt=OLD/'ground_truth'/scene/'ground_truth/gt.npz';cache=OLD/'correspondence'/(scene+'_no_repair.npz');before=sha(cache)
 ev=O/'evaluation'/variant/scene;ev.mkdir(parents=True,exist_ok=True)
 prov={'input_scope_verified':False,'scene':scene,'frame_count':200,'frame_list_sha256':hashlib.sha256(json.dumps(list(range(0,1000,5)),separators=(',',':')).encode()).hexdigest(),'selection_sha256':sha(OLD/'inputs'/scene/'selection.json'),'producer_receipt':str(root/'repair/commit_complete.json'),'producer_receipt_sha256':sha(root/'repair/commit_complete.json'),'prediction_sha256':sha(prediction),'GT_used_for_mapping':False}
 (ev/'source_provenance.json').write_text(json.dumps(prov,indent=2))
 state('RUNNING',stage='current_main_R2_before_DBSCAN')
 args=['adapt-surface','--gt',str(gt),'--surface',str(prediction),'--fixed-correspondence',str(cache),'--source-provenance',str(ev/'source_provenance.json'),'--method-name',variant,'--method-commit','source_sha256:'+sha(C/'source_lock.json'),'--out',str(ev/'r2')]
 with (ev/'adapt.log').open('a') as f:subprocess.run([sys.executable,'-B',str(OLDC/'evaluate_surface.py'),*args],cwd='/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert sha(cache)==before
 (ev/'complete.json').write_text(json.dumps({'status':'PASS','scene':scene,'variant':variant,'frames':200,'prediction':str(prediction),'prediction_sha256':sha(prediction),'same_xyz_rgb_and_correspondence_as_no_repair':True,'correspondence_sha256':before,'DBSCAN':False},indent=2))
 state('PASS',stage='FINAL_B_AND_R2_BEFORE_DBSCAN',DBSCAN_complete=False)
except Exception:state('FAILED',traceback=traceback.format_exc());raise
