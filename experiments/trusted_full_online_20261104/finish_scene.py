"""Publish approved final B once and score only the final published map."""
from alignment_common import *
import sys,subprocess
sys.path.insert(0,str(C/'policies/main'))
sys.path.insert(0,str(SRC))
from scene_completion.pipeline import write_scene_complete,manifest_from_commit,finalize
from revisable_instance_map.surface_evidence import reduce_surface_votes,CONFIRMED
FB=Path('/data/chenkejun/CVPR/results/ovimap_full_depth_baseline_20261101')
def protocol():
 lock=read(MAIN/'unified_eval/configs/current_protocol.json')
 assert lock['evaluable_GT_count']==350 and lock['profile_revision']==2 and lock['frozen'] is False
 assert sha(MAIN/'unified_eval/configs'/lock['config_file'])==lock['config_sha256']
 for name,h in lock['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/name)==h
 return lock
def run(args,log):
 with Path(log).open('w') as f:subprocess.run([sys.executable,'-B','-m','unified_eval.cli',*map(str,args)],cwd=MAIN,stdout=f,stderr=subprocess.STDOUT,check=True)
def finish(variant,scene):
 destination=O/'published'/variant/scene;destination.mkdir(parents=True,exist_ok=True)
 manifest=destination/'scene_complete.json'
 if not manifest.exists():
  if variant=='no_repair':
   src=FB/'scenes'/scene;frozen=read(FB/'final_audit.json')
   assert frozen['status']=='PASS' and sha(src/'instance_surface.npz')==frozen['prediction_sha256'][scene]
   assert read(src/'reuse_audit.json')['all_400_sample_caches_verified'] is True
   a=load(src/'instance_surface.npz');v=load(src/'strict_votes.npz');native=load(src/'strict_instance_surface.npz')
   assert list(v['frames'])==list(range(0,2000,5))
   assert str(v['xyz_sha256'])==digest(a['xyz_m'])
   np.testing.assert_array_equal(a['xyz_m'],native['xyz_m'])
   ev=reduce_surface_votes(v['keys'],v['votes'],len(a['xyz_m']),1<<32,2,.67)
   np.testing.assert_array_equal(native['instance_id'],np.where(ev['state']==CONFIRMED,ev['top1_instance_id'],-1))
   save(destination/'native.npz',**native,rgb=a['rgb'])
   save(destination/'surface_evidence.npz',xyz_m=a['xyz_m'],rgb=a['rgb'],**ev)
   write_scene_complete(manifest,scene=scene,workflow='no_repair',frame_ids=v['frames'].tolist(),inputs=dict(surface=src/'instance_surface.npz',native=destination/'native.npz',evidence=destination/'surface_evidence.npz',normal_geometry=O/'attributes'/scene/'400/normals.npz'),final_processing_complete=True)
  else:
   method,group=('main','A') if variant=='main' else ('online',variant[-1])
   root=O/method/scene/'400'/group
   m=manifest_from_commit(root,scene=scene,workflow='main' if method=='main' else 'online-v2')
   dump(manifest,m)
 receipt=finalize(manifest,destination/'final_B')
 assert receipt['status']=='PASS' and receipt['C_enabled'] is False and not receipt['online_state_updated']
 prediction=Path(receipt['final_surface']);a=load(prediction);attrs=load(O/'attributes'/scene/'400/normals.npz')
 np.testing.assert_array_equal(a['xyz_m'],attrs['xyz_m'])
 np.testing.assert_array_equal(a['rgb'],load(O/'attributes'/scene/'400/rgb.npz')['rgb'])
 lock=protocol();old=read(FB/'scenes'/scene/'evaluation/r2/adapter_manifest.json')
 ev=O/'evaluation'/variant/scene;ev.mkdir(parents=True,exist_ok=True)
 if not (ev/'complete.json').exists():
  run(['adapt-surface','--gt',old['gt_file'],'--surface',prediction,'--fixed-correspondence',old['correspondence_cache'],'--source-provenance',destination/'final_B/scene_finalization.json','--method-name','trusted_full_online_'+variant,'--method-commit',read(C/'source_lock.json')['run_id'],'--out',ev/'r2'],ev/'adapt.log')
  m=read(ev/'r2/adapter_manifest.json')
  invariants=['profile_config_sha256','evaluator_code_sha256','gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','geometry_xyz_sha256']
  for key in invariants:assert old[key]==m[key],key
  assert sha(old['correspondence_cache'])==sha(m['correspondence_cache'])
  dump(ev/'complete.json',dict(status='PASS',variant=variant,scene=scene,prediction=str(prediction),prediction_sha256=sha(prediction),final_B_receipt_sha256=sha(destination/'final_B/scene_finalization.json'),shared_correspondence_sha256=sha(old['correspondence_cache']),protocol_sha256=sha(MAIN/'unified_eval/configs/current_protocol.json')))
 print('FINAL_B_AND_R2_READY',variant,scene,receipt['newly_assigned'],flush=True)
if __name__=='__main__':finish(sys.argv[1],sys.argv[2])
