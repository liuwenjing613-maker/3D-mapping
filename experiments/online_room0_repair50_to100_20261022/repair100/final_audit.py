#!/usr/bin/env python3
"""Final, bounded read-only acceptance of the two authorized checkpoint-100 repairs."""
from pathlib import Path
from dataclasses import asdict
import sys, json, hashlib, time, re, subprocess, traceback, importlib.util
import numpy as np
HERE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/repair100')
sys.path.insert(0,str(HERE))
import pipeline as p
from revisable_instance_map.association import OnlineVoxelAssociator
C=HERE.parent
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
ROOT=TOP/'repair100'
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
EXPECTED_MAIN_HEAD='10c6dd44d28ea265eea5ace671c5c4370e932e48'
B=TOP/'prefix100'
STAGES=['direct','core_all','whole_to_partial']
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def equal(a,b,name):
 try:np.testing.assert_array_equal(a,b)
 except AssertionError as e:raise AssertionError(name+': '+str(e)[:800]) from e
def checks(d,names):
 for name in names:assert d.get(name) is True,name
def verify_hashes(hashes):
 for path,expected in hashes.items():assert sha(path)==expected,path
 return len(hashes)
def write(path,value):
 path=Path(path);assert path.is_relative_to(ROOT)
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def source_inheritance():
 previous=read(ROOT/'validation/layers_acceptance.json')
 assert previous['status']=='PASS'
 source=(HERE/'pipeline.py').read_text()
 assignment=' scene.settings=r.AssignmentSettings()\n'
 assert source.count(assignment)==1
 restored_hash=hashlib.sha256(source.replace(assignment,'').encode()).hexdigest()
 assert restored_hash==previous['actual_pipeline_sha256']
 defaults=asdict(p.r.AssignmentSettings())
 frozen50=read(TOP/'repair50/partial/direct/room0/complete.json')['postprocessing']['settings']
 assert defaults==frozen50
 return dict(status='PASS',validated_pipeline_sha256=previous['actual_pipeline_sha256'],
     final_pipeline_sha256=sha(HERE/'pipeline.py'),only_additional_line=assignment.strip(),
     removing_only_added_line_exactly_recovers_validated_source=True,
     default_postprocess_settings_equal_frozen50=True,
     layer50_and_baseline100_numerical_acceptance_inherited=True,
     numerical_report=str(ROOT/'validation/layers_acceptance.json'))

def source_freezes(arm):
 tr=TOP/'tracking100'/arm
 frozen=read(tr/'input_freeze.json')
 nbase=verify_hashes(frozen['original_baseline_source_hashes'])
 nseed=verify_hashes(frozen['seed_and_config_hashes'])
 assert sha(frozen['tracking_code_path'])==frozen['tracking_code_sha256']
 repair_freeze=read(ROOT/arm/'repair_input_freeze.json')
 nrepair=verify_hashes(repair_freeze['source_hashes'])
 assert repair_freeze['selection']=='original A13/B12 before95_90'
 expected=13 if arm=='A' else 12
 assert read(tr/'seed_manifest.json')['selected_masks']==expected
 validation=read(tr/'tracking_validation.json')
 assert validation['status']=='PASS'
 for case in validation['cases']:
  assert sha(tr/'cases'/case['case_uid']/'tracking/complete.json')==case['report_sha256']
 # Frozen tracking complete reports contain the original hashes of every PNG.
 label_files=0
 for case in read(tr/'seed_diagnostics.json')['cases']:
  report=read(tr/'cases'/case['case_uid']/'tracking/complete.json')
  assert report['completed_frames']==report['total_frames']==496
  for row in report['frames']:
   assert row['frame']<=495
   assert sha(row['label_file'])==row['label_sha256']
   label_files+=1
 return dict(status='PASS',original_seed_count=expected,baseline_files=nbase,seed_config_files=nseed,
     repair_inputs_including_gate_masks=nrepair,tracking_complete_reports=len(validation['cases']),
     tracking_label_files_byte_identical=label_files,tracking_gate_and_seed_sources_unchanged=True)

def arm_audit(arm,before):
 R=ROOT/arm
 complete=read(R/'repair_complete.json');commit=read(R/'commit_complete.json')
 assert complete['status']==commit['status']=='PASS'
 assert complete['group']==commit['group']==arm
 assert complete['raw_cutoff']==commit['last_raw_frame']==495
 assert complete['mapping_frames']==commit['processed_frames']==100
 assert complete['objects']==(13 if arm=='A' else 12)
 checks(complete,['source_hashes_unchanged','tracking_reused','gate_reused','main_mapping_paused'])
 assert complete['GT_used'] is False and commit['GT_used'] is False
 assert commit['state']=='COMMITTED_BRANCH_ONLY' and not commit['continuation_started']
 mapping=read(R/'partial/mapping_complete.json')
 assert mapping['status']=='PASS' and mapping['raw_RGB_cutoff']==495
 checks(mapping,['baseline_multilayer_candidates_votes_evidence_exact','full_replay_equals_delta',
                 'exact_rollback','strict_same_frame_point_vote','stages_committed_in_original_order'])
 parent=before['instance_id'];stage_rows=[]
 for stage in STAGES:
  F=R/'partial'/stage/'room0'
  after=load(F/'final/instance_surface.npz')
  for key in ['xyz_m','rgb']:equal(after[key],before[key],arm+' '+stage+' '+key)
  scope=load(F/'allowed_surface_ids.npz')['surface_point_index']
  assert len(scope)==len(np.unique(scope)) and np.all((scope>=0)&(scope<len(parent)))
  outside=np.ones(len(parent),bool);outside[scope]=False
  equal(after['instance_id'][outside],parent[outside],arm+' '+stage+' outside allowed scope')
  report=read(F/'complete.json')
  assert report['status']=='PASS' and report['scope_points']==len(scope)
  diff=int(np.sum(after['instance_id']!=parent))
  assert report['final_changed_vs_parent']==diff
  stage_rows.append(dict(stage=stage,geometry_and_rgb_exact=True,scope_points=len(scope),
                         changed_vs_parent=diff,outside_scope_changes=0))
  parent=after['instance_id']
 assert sha(commit['audit'])==commit['audit_sha256']
 assert sha(R/'repair_complete.json')==commit['repair_complete_sha256']
 audit=read(commit['audit'])
 checks(audit,['original50_entries_preserved_before100','every_old_entry_original_sample_voxels_exact',
   'unchanged_supports_preserved','roundtrip_checkpoint_exact','identity_counts_roundtrip_exact',
   'durable_sample_pixel_roundtrip_exact','surface_journal_reproduces_all_candidates_and_votes',
   'baseline_and_final_accumulated_votes_exact','delta_and_rollback_exact',
   'surface_evidence_fields_exact','native_labels_exact','strict_max_one_vote_per_frame_surface_point',
   'original_baseline_and_committed50_unchanged','main_mapping_paused'])
 for field in ['checkpoint','identity_counts','source_pixel_support','final_map','journal_manifest']:
  assert sha(commit[field])==commit[field+'_sha256'],field
 assert audit['old_map_version']==101 and audit['map_version']==commit['map_version']==102
 assert audit['old_identity_entries']==3226
 restored=OnlineVoxelAssociator.from_checkpoint(commit['checkpoint'])
 restored.identity_evidence.validate()
 assert restored.last_processed_frame_id==495 and restored.map_version==102
 assert restored.next_instance_id==commit['next_instance_id']
 assert len(restored.observation_support)==audit['identity_entries']
 before_checkpoint=OnlineVoxelAssociator.from_checkpoint(B/'association/observation_support_3cm.npz')
 assert restored.checkpoint_parameters()==before_checkpoint.checkpoint_parameters()
 del restored,before_checkpoint
 freeze=source_freezes(arm)
 return dict(status='PASS',original_seed_count=complete['objects'],stages=stage_rows,
     after_geometry_rgb_exact=True,strict_fullvote_delta_rollback_reports_pass=True,
     commit_checkpoint_roundtrip_report_pass=True,restored_checkpoint_validate_pass=True,
     map_version=102,next_instance_id=commit['next_instance_id'],identity_entries=audit['identity_entries'],
     old_entry_lineage_counts=audit['old_entry_lineage_counts'],
     unchanged_observations=audit['unchanged_observations'],
     revised_observations=audit['revised_observations'],added_observations=audit['added_observations'],
     after_map=commit['final_map'],after_map_sha256=commit['final_map_sha256'],
     input_freeze=freeze,commit_audit=str(Path(commit['audit'])),
     commit_audit_sha256=commit['audit_sha256'],committed_branch_only=True)

def io_audit():
 records=0;read_records=0;writes=0;probes=0;blocked=[];future=[];gt=[]
 mode_counts={};files={}
 pattern=re.compile(r'(?<![A-Za-z0-9])(?:frame|depth|f)(\d{6})(?!\d)')
 for path in sorted((ROOT/'io_audit').glob('*.jsonl')):
  files[str(path)]=sha(path)
  for line in path.read_text().splitlines():
   row=json.loads(line)
   if row.get('probe'):probes+=1;continue
   records+=1
   mode=str(row.get('mode',''));mode_counts[mode]=mode_counts.get(mode,0)+1
   if row.get('blocked'):blocked.append(row)
   is_write=any(c in mode for c in ['w','a','x']) and '+' not in mode
   if is_write:writes+=1;continue
   read_records+=1
   value=row.get('path','');name=Path(value).name
   match=pattern.search(name)
   if match and int(match.group(1))>495:future.append(row)
   if '/ground_truth/' in value or name in ['gt.npz','info_semantic.json','semantic.ply']:gt.append(row)
 assert files and records
 assert not blocked,blocked[:3]
 assert not future,future[:3]
 assert not gt,gt[:3]
 return dict(status='PASS',log_files=len(files),nonprobe_records=records,
     nonprobe_read_records=read_records,write_only_records_excluded_from_read_checks=writes,
     probe_records_excluded=probes,blocked_nonprobe=0,future_reads_over495=0,GT_reads=0,
     mode_counts=mode_counts,source_hashes=files)

def fixed_seed_comparison(before):
 O=TOP/'prefix100_new50/all_edge_20261021'
 grouping=read(O/'v1/grouping.json')
 seed=next(s for s in grouping['seeds'] if s['id']=='M0034')
 z=load(O/'v1/surface_supports.npz');idx=seed['index']
 support=z['indices'][z['offsets'][idx]:z['offsets'][idx+1]]
 assert len(support)==seed['support_count']==14336
 assert hashlib.sha256(support.tobytes()).hexdigest()==seed['support_sha256']
 assert seed['frame']==285
 cache=TOP/'prefix100_new50/overmerge/frame000285_visible.npz'
 upstream_path=TOP/'prefix100_new50/grouping_r2/grouping.json'
 assert sha(upstream_path)==grouping['source_sha256'][str(upstream_path)]
 upstream=read(upstream_path)
 assert sha(cache)==upstream['source_sha256'][str(cache)]
 visible=load(cache)['vertex_index']
 assert np.all(np.isin(support,visible))
 coverage=module('_final_coverage',C/'prefix100_new50/v2/coverage.py')
 original=next(x for x in read(O/'v2/measurements.json')['coverage'] if x['mask_id']=='M0034')
 reports={}
 surfaces={'before100':(B/'baseline/final/instance_surface.npz',B/'baseline/surface_evidence.npz')}
 for arm in ['A','B']:
  F=ROOT/arm/'partial/whole_to_partial/room0'
  surfaces[arm+'_after100']=(F/'final/instance_surface.npz',F/'native/surface_evidence.npz')
 for name,(surface_path,evidence_path) in surfaces.items():
  surface=load(surface_path);state=load(evidence_path)['state']
  equal(surface['xyz_m'],before['xyz_m'],name+' fixed seed geometry')
  labels=surface['instance_id'];ids,n=np.unique(labels[support],return_counts=True)
  histogram={str(int(i)):int(c) for i,c in zip(ids,n)}
  metrics=coverage.measure(support,visible,labels,state)
  target_intersection=int(np.sum(labels[support]==89));target_visible=int(np.sum(labels[visible]==89))
  target=dict(instance_id=89,intersection=target_intersection,instance_visible_count=target_visible,
      p=target_intersection/len(support),r=target_intersection/target_visible if target_visible else 0.)
  reports[name]=dict(current_ID_distribution=histogram,dominant_instance_measurement=metrics,
                    fixed_target89=target,surface_sha256=sha(surface_path))
  if name=='before100':
   for key in ['support_count','instance_id','intersection','instance_visible_count','p','r']:
    assert metrics[key]==original[key],key
 return dict(status='PASS',arm_seed='B/M0034',group_id='A003',source_raw_frame=285,
     source_processed_frame=58,fixed_support_points=14336,support_sha256=seed['support_sha256'],
     fixed_visible_points=len(visible),visible_cache_sha256=sha(cache),
     fixed_geometry=str(B/'baseline/final/instance_surface.npz'),
     meaning='same-domain map-label consistency against a fixed observed mask; not ground-truth quality or accuracy',
     p_definition='intersection / fixed mask support count14336',
     r_definition='intersection / current instance count in the same frozen source-visible set',
     GT_used=False,comparisons=reports)

def main():
 for arm in ['A','B']:
  for name in ['repair_complete.json','commit_complete.json']:
   path=ROOT/arm/name
   if not path.exists() or read(path).get('status')!='PASS':
    print(json.dumps(dict(status='NOT_READY',arm=arm,missing_or_incomplete=name)),flush=True);return
 started=time.perf_counter()
 inheritance=source_inheritance()
 before=load(B/'baseline/final/instance_surface.npz')
 # These are historical input locks, not new hashes taken only after repair.
 lock50=read(TOP/'repair50/input_freeze.json')['original_baseline_source_hashes']
 path50='/data/chenkejun/CVPR/results/online_room0_prefix50_20261021/baseline/final/instance_surface.npz'
 assert sha(path50)==lock50[path50]
 commit50=read(TOP/'committed50.json')
 assert sha(commit50['final_map'])==commit50['final_map_sha256']
 assert sha(commit50['checkpoint'])==commit50['checkpoint_sha256']
 arms={arm:arm_audit(arm,before) for arm in ['A','B']}
 io=io_audit()
 githead=subprocess.run(['git','-C',str(MAIN),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 gitstatus=subprocess.run(['git','-C',str(MAIN),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True).stdout
 assert githead==EXPECTED_MAIN_HEAD and not gitstatus.strip()
 p.verify_sources()
 compare=fixed_seed_comparison(before)
 report=dict(status='PASS',completed_unix=time.time(),seconds=time.perf_counter()-started,
   checkpoint_processed_frames=100,raw_cutoff=495,arms=arms,source_code_inheritance=inheritance,
   first50_source_maps_unchanged=dict(before_path=path50,before_sha256=lock50[path50],
      after_path=commit50['final_map'],after_sha256=commit50['final_map_sha256']),
   main_source=dict(HEAD=githead,tracked_worktree_clean=True,frozen_E1_and_R2_source_lock_pass=True),
   causality=io,B_M0034_fixed_support_comparison=compare,
   map_or_checkpoint_modified_by_validator=False,tracking_inference_or_repair_executed_by_validator=False)
 write(ROOT/'final_audit.json',report)
 print(json.dumps(dict(status='PASS',arms={a:dict(seed_count=arms[a]['original_seed_count'],after_geometry_rgb_exact=True,map_version=arms[a]['map_version']) for a in arms},
  causal_blocked=io['blocked_nonprobe'],main_source_unchanged=True,
  M0034={k:dict(distribution=v['current_ID_distribution'],target89=v['fixed_target89']) for k,v in compare['comparisons'].items()},
  report=str(ROOT/'final_audit.json'),seconds=report['seconds']),ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  write(ROOT/'final_audit_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()))
  raise
