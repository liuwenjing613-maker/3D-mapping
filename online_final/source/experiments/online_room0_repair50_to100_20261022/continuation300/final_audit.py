"""Independent acceptance for one branch/checkpoint, with no model or map mutation."""
from pathlib import Path
import os,sys,json,time,hashlib,re,subprocess,traceback
import numpy as np
from common_continue import CHECKPOINT,GROUP,ROOT,B,R as TRACK,CONT,BASEC,TOP,CURRENT_RAW,RAW_COUNT,FIDS,OLD_COUNT,MAIN
SNAPSHOT=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
sys.path.insert(0,str(SNAPSHOT/'revisable_instance_map/src'))
from revisable_instance_map.association import OnlineVoxelAssociator
REPAIR=ROOT/'repair'
def read(p):return json.loads(Path(p).read_text())
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,value):
 p=Path(p);assert p.is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def equal(a,b):np.testing.assert_array_equal(a,b)
def verify(hashes):
 for path,h in hashes.items():assert sha(path)==h,path
 return len(hashes)
def require_fields(x,names):
 for k in names:assert x.get(k) is True,k
def sources():
 frozen=read(TRACK/'input_freeze.json')
 counts=dict(baseline=verify(frozen['original_baseline_source_hashes']),seeds=verify(frozen['seed_and_config_hashes']))
 assert sha(frozen['tracking_code_path'])==frozen['tracking_code_sha256']
 counts['repair']=verify(read(REPAIR/'repair_input_freeze.json')['source_hashes'])
 manifest=read(TRACK/'seed_manifest.json');assert manifest['selected_masks']==read(REPAIR/'repair_complete.json')['objects']
 count=0
 for case in read(TRACK/'seed_diagnostics.json')['cases']:
  tr=read(TRACK/'cases'/case['case_uid']/'tracking/complete.json')
  assert tr['status']=='PASS' and tr['total_frames']==tr['completed_frames']==RAW_COUNT and not tr['GT_used']
  assert sorted(x['frame'] for x in tr['frames'])==list(range(RAW_COUNT))
  assert sha(case['config_path'])==tr['case_config_sha256']
  assert sha(TRACK/'cases'/case['case_uid']/'prepared_seed.png')==tr['seed_sha256']
  assert tr['directions_use_independent_seed_only_states']
  for row in tr['frames']:
   assert sha(row['label_file'])==row['label_sha256'];count+=1
  fid=case['source_choice']['frame'];assert fid in FIDS
  if GROUP=='B':assert fid>=OLD_COUNT*5
 counts.update(selected_masks=manifest['selected_masks'],tracking_files=count,source_windows_valid=True)
 return counts
def causality():
 pattern=re.compile(r'(?<![A-Za-z0-9])(?:frame|depth|f)(\d{6})(?=$|[._-])')
 rows=reads=writes=probes=0;bad=[];files={}
 roots=[ROOT/'io_audit',B/'io_audit']
 for folder in roots:
  for path in sorted(folder.glob('*.jsonl')):
   files[str(path)]=sha(path)
   with path.open() as f:
    for line in f:
     x=json.loads(line)
     if x.get('probe'):probes+=1;continue
     rows+=1;mode=str(x.get('mode',''))
     if x.get('blocked'):bad.append(x)
     if any(c in mode for c in 'wax') and '+' not in mode:writes+=1;continue
     reads+=1;value=x.get('path','');name=Path(value).name;match=pattern.search(name)
     # ROI hash names are not frame IDs. Only canonical raw image/journal names count.
     if name.startswith(('ROI','roi')):match=None
     if match and int(match.group(1))>min(CURRENT_RAW,x.get('cutoff',CURRENT_RAW)):bad.append(x)
     if '/ground_truth/' in value or name in ['gt.npz','info_semantic.json','semantic.ply']:bad.append(x)
 assert files and rows
 assert not bad,bad[:3]
 return dict(status='PASS',log_files=len(files),records=rows,reads=reads,writes=writes,probes=probes,future_reads=0,GT_reads=0,blocked=0,source_hashes=files)
def execution_provenance():
 result={}
 for stage,folder in [('mapping',B/'execution_code'),('commit',REPAIR/'execution_code')]:
  path=folder/'manifest.json';m=read(path);assert m['status']=='CAPTURED'
  for row in m['files']:
   assert sha(row['snapshot'])==row['sha256'],row['snapshot']
   assert sha(row['source'])==row['current_source_sha256_at_capture'],row['source']
  detail=dict(manifest=str(path),manifest_sha256=sha(path),prelaunch=m['prelaunch'],snapshot_files_exact=True,
    source_files_unchanged_since_capture=True,files=len(m['files']))
  delta=folder/'retrospective_helper_delta.json'
  if delta.exists():
   d=read(delta);assert d['status']=='DOCUMENTED_DIFFERENCE'
   detail.update(retrospective_helper_delta=str(delta),delta_sha256=sha(delta),
    limitation='Mapping200 helper source recovered after launch from session text; this is explicitly not a prelaunch snapshot.')
  result[stage]=detail
 return result
def selection_protocol():
 path=TOP/'continuation300/selection_protocol_audit.json';data=read(path)
 rows=[x for x in data['rows'] if x['checkpoint']==CHECKPOINT and x['group']==GROUP]
 assert len(rows)==1
 row=rows[0];assert row['status']=='PASS'
 assert row['candidate_processed_range']==[1 if GROUP=='A' else OLD_COUNT+1,CHECKPOINT]
 assert row['dense_tracking_range']==[0,CURRENT_RAW] and row['dense_RGB_count']==RAW_COUNT
 assert row['scoring_history']==[1,CHECKPOINT] and row['previous_checkpoint']==OLD_COUNT
 assert row['retained']==read(REPAIR/'repair_complete.json')['objects']
 require_fields(row,['all_primary_and_aux_mask_source_frames_valid','original_seed_mask_sha256_preserved_for_tracking','both_independent_directions_full_arrived_history','map_source_branch_correct'])
 boundary=row['boundary_entry_audit'];assert boundary['status']=='PASS'
 require_fields(boundary,['full_frame_and_all_alias_crop_edges_recomputed','edge_masks_removed_before_noise_and_regrouping','survivor_partition_exact'])
 snapshot=ROOT/'selection_protocol_snapshot.json'
 write(snapshot,dict(status='PASS',source_file=str(path),source_sha256_at_read=sha(path),row=row))
 return dict(snapshot=str(snapshot),snapshot_sha256=sha(snapshot),status='PASS',retained=row['retained'],boundary_counts=boundary['counts'])
def main():
 started=time.perf_counter()
 before=load(B/'baseline/final/instance_surface.npz')
 done=read(REPAIR/'repair_complete.json');commit=read(REPAIR/'commit_complete.json')
 assert done['status']==commit['status']=='PASS'
 assert done['group']==commit['group']==GROUP
 assert done['raw_cutoff']==commit['last_raw_frame']==CURRENT_RAW
 assert done['mapping_frames']==commit['processed_frames']==CHECKPOINT
 assert not done['GT_used'] and not commit['GT_used']
 require_fields(done,['source_hashes_unchanged','main_mapping_paused'])
 repair=read(REPAIR/'partial/mapping_complete.json')
 require_fields(repair,['baseline_multilayer_candidates_votes_evidence_exact','full_replay_equals_delta','exact_rollback','strict_same_frame_point_vote','stages_committed_in_original_order'])
 parent=before['instance_id'];stages=[]
 for stage in ('direct','core_all','whole_to_partial'):
  f=REPAIR/'partial'/stage/'room0';after=load(f/'final/instance_surface.npz')
  for k in ['xyz_m','rgb']:equal(after[k],before[k])
  ids=load(f/'allowed_surface_ids.npz')['surface_point_index']
  assert len(ids)==len(np.unique(ids)) and np.all((ids>=0)&(ids<len(parent)))
  outside=np.ones(len(parent),bool);outside[ids]=False;equal(after['instance_id'][outside],parent[outside])
  report=read(f/'complete.json');change=int((after['instance_id']!=parent).sum())
  assert report['scope_points']==len(ids) and report['final_changed_vs_parent']==change
  stages.append(dict(stage=stage,scope_points=len(ids),changed_vs_parent=change,outside_scope_changes=0))
  parent=after['instance_id']
 for field in ('checkpoint','identity_counts','source_pixel_support','observation_lineage','final_map','journal_manifest','audit'):
  assert sha(commit[field])==commit[field+'_sha256'],field
 assert sha(REPAIR/'repair_complete.json')==commit['repair_complete_sha256']
 audit=read(commit['audit']);assert audit['status']=='PASS'
 old=OnlineVoxelAssociator.from_checkpoint(B/'association/observation_support_3cm.npz')
 new=OnlineVoxelAssociator.from_checkpoint(commit['checkpoint']);new.identity_evidence.validate()
 assert new.last_processed_frame_id==CURRENT_RAW and new.map_version==old.map_version+1==commit['map_version']
 assert new.next_instance_id==commit['next_instance_id'] and new.checkpoint_parameters()==old.checkpoint_parameters()
 prior=read(B/'prior_commit.json');assert prior['processed_frames']==OLD_COUNT and prior['group']==GROUP
 assert sha(prior['checkpoint'])==prior['checkpoint_sha256']
 journal=read(commit['journal_manifest']);assert sorted(x['frame_id'] for x in journal['frames'])==FIDS
 oldjournal=read(prior['journal_manifest']);oldlayers={x['frame_id']:len(x['layer_files']) for x in oldjournal['frames']}
 for row in journal['frames']:
  assert row['old_layer_count']==oldlayers.get(row['frame_id'],1)
  assert len(row['layer_files'])==row['old_layer_count']+1
  for x in row['layer_files']:assert sha(x['path'])==x['sha256']
 original=read(B/'mapping_complete.json');assert original['frame_count']==CHECKPOINT and original['status']=='PASS'
 frozen=sources()
 githead=subprocess.run(['git','-C',str(MAIN),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 assert githead=='10c6dd44d28ea265eea5ace671c5c4370e932e48'
 assert not subprocess.run(['git','-C',str(MAIN),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True).stdout.strip()
 code=read(CONT/'repair_adapter_audit.json')
 assert sha(CONT/'repair_continue.py')==code['source_sha256']
 assert sha(BASEC/'repair100/pipeline.py')==code['reference_sha256']
 report=dict(status='PASS',checkpoint=CHECKPOINT,group=GROUP,raw_cutoff=CURRENT_RAW,seconds=time.perf_counter()-started,
   stages=stages,source_freezes=frozen,selection_protocol=selection_protocol(),causality=causality(),execution_code=execution_provenance(),main_HEAD=githead,map_version=new.map_version,
   identity_entries=len(new.observation_support),next_instance_id=new.next_instance_id,
   prior_checkpoint_used=prior['checkpoint'],prior_checkpoint_sha256=prior['checkpoint_sha256'],
   dynamic_layer_lineage_validated=True,geometry_and_RGB_unchanged=True,checkpoint_restore_validated=True,
   commit_audit=commit['audit'],commit_audit_sha256=commit['audit_sha256'],
   assigned_before=int((before['instance_id']>0).sum()),assigned_after=int((parent>0).sum()),
   labels_changed_before_after=int((before['instance_id']!=parent).sum()),GT_used=False,
   meaning='same observed geometry and unchanged fields; coverage is not ground-truth accuracy')
 report['seconds']=time.perf_counter()-started
 write(ROOT/'final_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['causality','source_freezes','stages']}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:write(ROOT/'final_audit_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()));raise
