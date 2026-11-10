"""Use the current terminal numerical rules with real single-repair lineage.
No repair commit is invented at 400; the finalized mapper supplies raw candidate votes.
"""
from pathlib import Path
import ast,hashlib,json,os,re,sys,time,types,traceback,fcntl
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110')
SC=Path('/home/chenkejun/CVPR/experiments/unified_full_depth_20261101');SO=Path('/data/chenkejun/CVPR/results/online_final_trusted_20261110')
BASE=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101')
BASELINE_CONTROL='--baseline-control' in sys.argv
if BASELINE_CONTROL:SO=BASE;O=O/'baseline_terminal_control'
SRC=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src')
sys.path.insert(0,str(SRC));sys.path.insert(0,str(SC/'policies/main'))
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.frame_io import ReplicaFrameSource
from revisable_instance_map.observations import RawInstanceObservation,source_pixel_indices
from revisable_instance_map.surface_evidence import reduce_surface_votes
from scene_completion.core import complete
from scene_completion.pipeline import write_scene_complete,finalize
GUARD=SC/'ablations/dbscan_identity_completion_guarded_20261102/run_guarded_covisible.py'
HISTORY=SC/'ablations/dbscan_identity_completion_guarded_20261102/run_history_covisible.py'
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,**d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.stem+'.tmp.npz');np.savez_compressed(tmp,**d);tmp.replace(p)
def function_text(path,name):
 text=path.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.get_source_segment(text,node)+'\n'
def cfg(path):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CFG' for t in n.targets))
 return eval(compile(ast.Expression(node.value),str(path),'eval'),{'dict':dict})
# Import precisely the original sample function without executing its repair driver.
exec(compile(function_text(SC/'online/room1/commit_continue.py','sample_pixels'),str(SC/'online/room1/commit_continue.py'),'exec'),globals())
def export_pixels(scene,out):
 dest=out/'source_pixel_support.npz'
 if dest.exists():return dest
 root=O/scene/'400/B/mapping';prior=read(root/'prior_commit.json');old=load(prior['source_pixel_support'])
 old_index={str(v):i for i,v in enumerate(old['observation_ids'])}
 assoc=OnlineVoxelAssociator.from_checkpoint(root/'association/observation_support_3cm.npz');assoc.identity_evidence.validate()
 source=ReplicaFrameSource(root/'config.json');assert list(source.frame_ids)==list(range(0,2000,5))
 originals={x['observation_id']:RawInstanceObservation(**x) for x in [json.loads(line) for line in (root/'association/observations.jsonl').read_text().splitlines()]}
 entries=sorted(assoc.observation_support.values(),key=lambda x:(x.frame_id,x.observation_id));pix=[];vox=[];layers=[];ids=[];fids=[];pids=[];frame=None;fid=None
 for entry in entries:
  if entry.frame_id<=245:
   i=old_index[entry.observation_id];sl=slice(old['offsets'][i],old['offsets'][i+1]);p=old['flat_source_pixel_index'][sl];v=old['sampled_voxel_coordinates'][sl];layer=int(old['layer_index'][i])
   assert int(old['instance_ids'][i])==entry.instance_id and int(old['frame_ids'][i])==entry.frame_id
  else:
   if fid!=entry.frame_id:fid=entry.frame_id;frame=source.load_frame(fid)
   ob=originals[entry.observation_id];assert ob.source_mask_sha256==entry.source_mask_sha256
   p,v=sample_pixels(frame,ob,assoc);layer=0
  assert tuple(tuple(int(x) for x in row) for row in np.unique(v,axis=0))==entry.voxels,entry.observation_id
  pix.append(p);vox.append(v);layers.append(layer);ids.append(entry.observation_id);fids.append(entry.frame_id);pids.append(entry.instance_id)
 save(dest,observation_ids=np.asarray(ids,dtype=str),frame_ids=np.asarray(fids,np.int32),instance_ids=np.asarray(pids,np.int32),layer_index=np.asarray(layers,np.int16),
  offsets=np.r_[0,np.cumsum([len(p) for p in pix],dtype=np.int64)],flat_source_pixel_index=np.concatenate(pix).astype(np.int32),sampled_voxel_coordinates=np.concatenate(vox).astype(np.int32),voxel_size_m=np.asarray([assoc.voxel_size_m],np.float64))
 dump(out/'pixel_export_audit.json',dict(status='PASS',source_checkpoint=str(root/'association/observation_support_3cm.npz'),source_checkpoint_sha256=sha(root/'association/observation_support_3cm.npz'),
  actual_entries=len(entries),retained_prior_entries=len(old_index),new_original_entries=sum(f>245 for f in fids),all_sampled_voxels_exact=True,no_new_repair_commit=True,next_instance_id=assoc.next_instance_id))
 return dest

def module(scene,out,control=False):
 g=types.ModuleType('current_terminal_numerics')
 g.__dict__.update(Path=Path,np=np,os=os,re=re,time=time,cKDTree=cKDTree,DBSCAN=DBSCAN,read=read,load=load,save=save,dump=dump,sha=sha,O=SO,CFG=cfg(GUARD),
  HISTORY_ROOT=SO/'online'/scene,ACTUAL_CHECKPOINTS=[50,100,200,300,400] if control else [50],TRANSACTIONS=SO/'online'/scene/'400/B/repair/partial/transactions' if control else out/'transactions')
 for name in ['strict','encode','voxel_reps','build_candidates','replay_votes','fill_gray']:
  s=function_text(GUARD,name)
  if name=='build_candidates':
   s=s.replace("branch=Path(pixel).parents[2];scene_root=branch.parents[1];group=branch.name", "scene_root=HISTORY_ROOT;group='B'")
   s=s.replace('for cp in [50,100,200,300,400]:','for cp in ACTUAL_CHECKPOINTS:')
  if name=='replay_votes':s=s.replace("(root/'repair/partial/transactions').glob('f*.npz')","TRANSACTIONS.glob('f*.npz')")
  exec(compile(s,str(GUARD),'exec'),g.__dict__)
 return g

def run(scene,control=False):
 out=O/('control_replay' if control else 'postprocess')/scene;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists() and read(out/'complete.json').get('status')=='PASS':return read(out/'complete.json')
 started=time.time();root=SO/'online'/scene/'400/B' if control else O/scene/'400/B';mapping=root/'mapping'
 assert read(mapping/'mapping_complete.json')['status']=='PASS';g=module(scene,out,control)
 if control:
  manifest_path=SO/'published/online_B'/scene/'scene_complete.json';m=read(manifest_path);pixel=root/'repair/commit_candidate/source_pixel_support.npz'
  src=load(m['inputs']['surface']);evidence=load(m['inputs']['evidence']);pair=load(Path(m['inputs']['evidence']).with_name('surface_instance_frame_votes.npz'))
  current_core=load(root/'repair/e1/region.npz')['core_owner'];base=read(root/'repair/global_association.json')['instance_base']
  original_path=SO/'published/online_B'/scene/'final_B/instance_surface.npz'
 else:
  assert read(O/scene/'mapping_audit.json')['status']=='PASS';pixel=export_pixels(scene,out)
  src=load(mapping/'baseline/final/instance_surface.npz');evidence=load(mapping/'baseline/surface_evidence.npz');pair=load(mapping/'baseline/surface_instance_frame_votes.npz')
  base=max(int(pair['instance_id'].max())+1,read(out/'pixel_export_audit.json')['next_instance_id']);current_core=np.full(len(src['xyz_m']),-1,np.int32)
  manifest_path=out/'scene_complete.json'
  if not manifest_path.exists():
   write_scene_complete(manifest_path,scene=scene,workflow='online-v2',frame_ids=list(range(0,2000,5)),inputs=dict(surface=mapping/'baseline/final/instance_surface.npz',native=mapping/'baseline/native/instance_surface.npz',evidence=mapping/'baseline/surface_evidence.npz',normal_geometry=SO/'attributes'/scene/'400/normals.npz'),final_processing_complete=True,final_repair_complete=True)
   m=read(manifest_path);m.update(actual_repair_schedule=[50],last_repair_checkpoint=50,final_400_repair_executed=False,producer_mapping_audit=str(O/scene/'mapping_audit.json'));dump(manifest_path,m)
  m=read(manifest_path);receipt=finalize(manifest_path,out/'final_B');original_path=Path(receipt['final_surface'])
 original=load(original_path);xyz=src['xyz_m'];n=len(xyz);attrs=load(SO/'attributes'/scene/'400/normals.npz');rgb=load(SO/'attributes'/scene/'400/rgb.npz');norm=attrs['normals'];valid=attrs['normal_valid']&rgb['color_valid']
 np.testing.assert_array_equal(xyz,attrs['xyz_m']);np.testing.assert_array_equal(src['rgb'],rgb['rgb'])
 old=reduce_surface_votes(g.encode(pair,base),pair['frame_votes'],n,base,2,.67)
 for k,value in old.items():np.testing.assert_array_equal(value,evidence[k])
 ctl,_,_=complete(xyz,norm,src['instance_id'],old);np.testing.assert_array_equal(ctl,original['instance_id']);del ctl,evidence
 basecase=out/'base';basecase.mkdir(exist_ok=True)
 if not (basecase/'complete.json').exists():
  lut,decisions=g.build_candidates(xyz,original['instance_id'],pair,pixel,basecase)
  if len(lut)<base:lut=np.r_[lut,np.arange(len(lut),base,dtype=np.int32)]
  if decisions['families']:
   if not control:
    g.TRANSACTIONS.mkdir(exist_ok=True)
    for fid in range(0,2000,5):
     s=load(mapping/'baseline/frame_support'/f'f{fid:06d}.npz');raw=np.unique(s['surface_point_index'].astype(np.int64)*base+s['candidate_persistent_id']);keys=g.strict(raw,base)
     save(g.TRANSACTIONS/f'f{fid:06d}.npz',whole_to_partial_candidate_keys=raw,whole_to_partial_frame_keys=keys)
   kk,vv,mp=g.replay_votes(root,pair,base,lut,basecase);merged=reduce_surface_votes(kk,vv,n,base,2,.67)
  else:
   merged=old;mp=pair;dump(basecase/'frame_replay_audit.json',dict(status='PASS',no_identity_changes=True,original_votes_reused=True))
  lab=src['instance_id'].copy();assigned=lab>0;lab[assigned]=lut[lab[assigned]];was=old['state']==2;now=merged['state']==2;lab[was&~now]=-1;lab[now]=merged['top1_instance_id'][now]
  mergeout=basecase/'merge_only';mergeout.mkdir(exist_ok=True)
  save(mergeout/'pre_B.npz',xyz_m=xyz,rgb=src['rgb'],instance_id=lab);save(mergeout/'native.npz',xyz_m=xyz,rgb=src['rgb'],instance_id=np.where(now,merged['top1_instance_id'],-1).astype(np.int32));save(mergeout/'evidence.npz',xyz_m=xyz,rgb=src['rgb'],**merged)
  if not (mergeout/'scene_complete.json').exists():write_scene_complete(mergeout/'scene_complete.json',scene=scene,workflow='online-v2',frame_ids=list(range(0,2000,5)),inputs=dict(surface=mergeout/'pre_B.npz',native=mergeout/'native.npz',evidence=mergeout/'evidence.npz',normal_geometry=SO/'attributes'/scene/'400/normals.npz'),final_processing_complete=True,final_repair_complete=True)
  receipt=finalize(mergeout/'scene_complete.json',mergeout/'final_B');merged_labels=load(receipt['final_surface'])['instance_id']
  bothout=basecase/'merge_fill';bothout.mkdir(exist_ok=True);both=g.fill_gray(xyz,norm,valid,merged_labels,merged,mp,bothout)
  save(bothout/'instance_surface.npz',xyz_m=xyz,rgb=src['rgb'],instance_id=both);dump(bothout/'complete.json',dict(status='PASS',temporary_metrics={},label_decisions_frozen_without_GT=True));dump(basecase/'complete.json',dict(status='PASS',families=decisions['families']))
 else:
  decisions=read(basecase/'merge_decisions.json');assert decisions['settings']==g.CFG
  both=load(basecase/'merge_fill/instance_surface.npz')['instance_id']
 def emit(v,s,arm,original,labels,target,provenance):
  save(target/'instance_surface.npz',xyz_m=original['xyz_m'],rgb=original['rgb'],instance_id=labels);dump(target/'provenance.json',provenance)
  return dict(status='FROZEN_BEFORE_EVALUATION',prediction_sha256=sha(target/'instance_surface.npz'),temporary_prediction=str(target/'instance_surface.npz'),history_checkpoints_used=[50] if not control else [50,100,200,300])
 g.evaluate=emit
 namespace=dict(__file__=str(HISTORY),Path=Path,np=np,time=time,re=re,cKDTree=cKDTree,DBSCAN=DBSCAN,CFG=cfg(HISTORY),g=g,O=SO,
  CURRENT_MANIFEST=manifest_path,CURRENT_FINAL_B=original_path,HISTORY_ROOT=SO/'online'/scene,PIXEL_PATH=pixel,CURRENT_CORE=current_core)
 s=function_text(HISTORY,'core_completion')
 s=s.replace("manifest=g.read(O/'published'/v/s/'scene_complete.json')","manifest=g.read(CURRENT_MANIFEST)")
 s=s.replace("original=g.load(O/'published'/v/s/'final_B/instance_surface.npz')","original=g.load(CURRENT_FINAL_B)")
 s=s.replace("branch=O/('main' if v=='main' else 'online')/s","branch=HISTORY_ROOT")
 s=s.replace("root=branch/'400'/group;pixel=g.load(root/'repair/commit_candidate/source_pixel_support.npz')","pixel=g.load(PIXEL_PATH)")
 s=s.replace("current=g.load(root/'repair/e1/region.npz')['core_owner']","current=CURRENT_CORE.copy()")
 # A repair may remove current votes while preserving spatial labels outside its region.
 # Validate provenance of such retained labels before history completion.
 history_pair=load(basecase/'merged_votes.npz') if (basecase/'merged_votes.npz').exists() else pair
 unsupported=(both>0)&~np.isin(both,history_pair['instance_id'])
 support_audit=dict(status='PASS',retained_without_current_vote_points=int(unsupported.sum()),retained_ids=np.unique(both[unsupported]).tolist(),new_assignments_require_current_votes=True)
 if np.any(unsupported):
  before_repair=load(mapping/'baseline/final/instance_surface.npz')
  region=load(root/'repair/e1/region.npz')
  np.testing.assert_array_equal(before_repair['xyz_m'],xyz)
  np.testing.assert_array_equal(both[unsupported],original['instance_id'][unsupported])
  np.testing.assert_array_equal(both[unsupported],src['instance_id'][unsupported])
  np.testing.assert_array_equal(both[unsupported],before_repair['instance_id'][unsupported])
  assert np.all(region['owner'][unsupported]<=0)
  assert np.all(region['core_owner'][unsupported]<=0)
  support_audit['provenance']='Exact pre-repair labels outside current repair region; unchanged in committed surface, finalB and base DBSCAN.'
 dump(out/'retained_label_provenance_audit.json',support_audit)
 old_assert="assert np.all(np.isin(result[result>0],pairids))"
 assert s.count(old_assert)==1
 s=s.replace(old_assert,"assert np.all(np.isin(result[(result>0)&(labels<=0)],pairids));assert np.all(np.isin(result[result>0],np.union1d(labels[labels>0],pairids)))")
 exec(compile(s,str(HISTORY),'exec'),namespace);result=namespace['core_completion']('online_B',scene,basecase,out)
 history=read(out/'decision_freeze.json');active=set(map(int,history['active_history']))
 assert active<=({50,100,200,300,400} if control else {50}),active
 actual=load(out/'instance_surface.npz');assert np.array_equal(actual['instance_id'][both>0],both[both>0]);np.testing.assert_array_equal(actual['xyz_m'],xyz)
 if control and BASELINE_CONTROL:
  expected=load(SO/'ablations/dbscan_history_covisible_20261102/online_B'/scene/'instance_surface.npz')
  for key in ['xyz_m','rgb','instance_id']:np.testing.assert_array_equal(actual[key],expected[key])
 result.update(status='PASS',scene=scene,control_replay=control,control_exact=bool(control and BASELINE_CONTROL),seconds=time.time()-started,actual_repair_checkpoints=sorted(active),
  merge_families=decisions['families'],current_repair_core_present=control,historical_completion_added=history['added_gray'],
  numerical_source_sha256={str(GUARD):sha(GUARD),str(HISTORY):sha(HISTORY)},original_mapping_checkpoint_unchanged=True)
 dump(out/'complete.json',result);print('DBSCAN_PASS',scene,'control' if control else 'single',result['seconds'],flush=True);return result
if __name__=='__main__':
 scene=sys.argv[1];control=True
 try:
  lockroot=O/('control_replay' if control else 'postprocess')/scene;lockroot.mkdir(parents=True,exist_ok=True)
  with (lockroot/'run.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);run(scene,control)
 except Exception:
  dump(O/('control_replay' if control else 'postprocess')/scene/'failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
