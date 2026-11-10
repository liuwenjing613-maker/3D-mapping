#!/usr/bin/env python3
"""Independent read-only regression for the checkpoint-100 layer adapter.
Only this validator's reports are written; neither repair pipeline nor commit is invoked.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,time,hashlib,traceback,os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pipeline as p
C=HERE.parent
TOP=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
OUT=TOP/'repair100/validation'
STAGES=('direct','core_all','whole_to_partial')

def read(path):return json.loads(Path(path).read_text())
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(name,data):
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def strict_oracle(keys,base):
 # Independent point-ID table oracle; duplicate same-ID evidence does not add a vote.
 keys=np.unique(keys.astype(np.int64))
 pair=np.column_stack((keys//base,keys%base))
 pt,n=np.unique(pair[:,0],return_counts=True)
 chosen=pair[np.isin(pair[:,0],pt[n==1])]
 return np.sort(chosen[:,0]*base+chosen[:,1])
def aggregate(parts):return np.unique(np.concatenate(parts),return_counts=True)
def pair_keys(z,base):
 keys=z['surface_point_index'].astype(np.int64)*base+z['instance_id']
 order=np.argsort(keys)
 return keys[order],z['frame_votes'][order]
def equal(actual,expected,label):
 try:np.testing.assert_array_equal(actual,expected)
 except AssertionError as e:raise AssertionError(label+': '+str(e)[:1200]) from e
def stage_report(stage,frames,seconds):print(json.dumps(dict(stage=stage,frames=frames,seconds=seconds)),flush=True)

def check50():
 started=time.perf_counter()
 R=TOP/'repair50'
 B=Path('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021')
 surface=load(B/'baseline/native/instance_surface.npz')
 source=p.r.ReplicaFrameSource(C/'mapping_config50.json')
 assert list(source.frame_ids)==list(range(0,246,5))
 scene=SimpleNamespace(xyz=surface['xyz_m'],tree=cKDTree(surface['xyz_m'].astype(np.float64)))
 lookup={(row['frame_id'],row['mask_local_id']):row['instance_id'] for row in map(json.loads,(B/'association/associations.jsonl').read_text().splitlines())}
 association=read(R/'global_association.json');base=association['instance_base']
 canonical={tid:x['canonical_track_id'] for x in association['objects'] for tid in x['member_track_ids']}
 fixed={x['canonical_track_id']:x['persistent_id'] for x in association['objects']}
 obj={};info={}
 for case in read(R/'seed_diagnostics.json')['cases']:
  uid=case['case_uid'];info[uid]={'config':read(case['config_path'])}
  for item in case['objects']:obj[item['track_id']]=uid
 decisions={row['frame']:row for row in read(R/'validated/full_track/frame_decisions.json')['frames']}
 region=load(R/'e1/region.npz');prior,owner=region['prior_owner'],region['owner']
 apoints=np.flatnonzero((prior!=-1)|(owner!=-1))
 to_pid=np.zeros(max(fixed)+1,np.uint16)
 for cid,pid in fixed.items():to_pid[cid]=pid
 all_frames={s:[] for s in ('baseline',*STAGES)}
 stage_scopes={s:[] for s in STAGES}
 comparison_counts=dict(baseline_candidate_arrays=0,stage_candidate_arrays=0,stage_effective_arrays=0,
                        full_pixel_history_images=0,full_pixel_adopted_images=0,full_pixel_residual_images=0)
 for index,fid in enumerate(source.frame_ids,1):
  frame=source.load_frame(fid)
  table=np.zeros(int(frame.mask_local.max())+1,np.uint16)
  for mid in np.unique(frame.mask_local):
   if mid:table[mid]=lookup[(fid,int(mid))]
  layer=table[frame.mask_local]
  merged=np.asarray(Image.open(R/'validated/full_track/accepted_masks'/f'f{fid:06d}.png'),np.uint16)
  eligible=p.family_targets(decisions[fid]['accepted_track_ids'],obj,info,canonical)
  residual=p.partial_layer(layer,merged,eligible)
  expected_residual=table[np.asarray(Image.open(R/'commit_candidate/pixel_journal'/f'f{fid:06d}_residual_local.png'))]
  equal(residual,expected_residual,f'50 raw{fid} residual pixels')
  comparison_counts['full_pixel_residual_images']+=1
  visible=np.zeros(len(scene.xyz),bool)
  visible[apoints],_,_=p.obs.visible_samples(scene.xyz[apoints],frame.camera_to_world,frame.camera,frame.depth_m,.02)
  actual=p.transform_layer(frame,layer,merged,eligible,scene,prior,owner,visible,base)
  tx=load(R/'partial/transactions'/f'f{fid:06d}.npz')
  equal(actual['before'],tx['baseline_candidate_keys'],f'50 raw{fid} baseline candidates')
  comparison_counts['baseline_candidate_arrays']+=1
  baseline=strict_oracle(actual['before'],base)
  equal(baseline,tx['baseline_frame_keys'],f'50 raw{fid} baseline effective')
  all_frames['baseline'].append(baseline)
  np_,nc=p.h.project(frame,merged,scene)
  new=p.h.keys(np_,to_pid[nc],base)
  direct_scope=np.unique(np.r_[actual['retired'],np_])
  assert np.all(np.isin(direct_scope,np_))
  previous_effective=baseline
  for stage in STAGES:
   candidates=np.union1d(actual['candidates'][stage],new)
   equal(candidates,tx[stage+'_candidate_keys'],f'50 raw{fid} {stage} candidates')
   effective=strict_oracle(candidates,base)
   equal(effective,tx[stage+'_frame_keys'],f'50 raw{fid} {stage} effective')
   all_frames[stage].append(effective)
   stage_scopes[stage].append(direct_scope if stage=='direct' else np.unique(np.setxor1d(previous_effective,effective)//base))
   previous_effective=effective
   comparison_counts['stage_candidate_arrays']+=1;comparison_counts['stage_effective_arrays']+=1
  equal(actual['journal'],np.asarray(Image.open(R/'commit_candidate/pixel_journal'/f'f{fid:06d}_history.png')),f'50 raw{fid} history pixels')
  equal(to_pid[merged],np.asarray(Image.open(R/'commit_candidate/pixel_journal'/f'f{fid:06d}_adopted.png')),f'50 raw{fid} adopted pixels')
  comparison_counts['full_pixel_history_images']+=1;comparison_counts['full_pixel_adopted_images']+=1
  if index%10==0:stage_report('actual_pipeline_50_equivalence',index,time.perf_counter()-started)
 reports={}
 previous_pair=None
 for stage in ('baseline',*STAGES):
  folder=B/'baseline' if stage=='baseline' else R/'partial'/stage/'room0/native'
  k,v=aggregate(all_frames[stage])
  target=load(folder/'surface_instance_frame_votes.npz')
  ek,ev=pair_keys(target,base);equal(k,ek,stage+' aggregate keys');equal(v,ev,stage+' aggregate votes')
  evidence=p.r.reduce_surface_votes(k,v,len(scene.xyz),base,2,.67)
  expected_evidence=load(folder/'surface_evidence.npz')
  for field in p.r.FIELDS:equal(evidence[field],expected_evidence[field],stage+' reduced evidence '+field)
  if stage!='baseline':
   scope=np.unique(np.concatenate(stage_scopes[stage]))
   if stage=='whole_to_partial':
    pk,pv=previous_pair
    common,ni,pi=np.intersect1d(k,pk,return_indices=True)
    scope=np.unique(np.r_[np.setxor1d(k,pk)//base,common[v[ni]!=pv[pi]]//base])
   expected_scope=load(R/'partial'/stage/'room0/allowed_surface_ids.npz')['surface_point_index']
   equal(scope,expected_scope,stage+' allowed commit scope')
  previous_pair=(k,v)
  reports[stage]=dict(keys=len(k),votes=int(v.sum()),vote_arrays_exact=True,commit_scope_exact=stage!='baseline',evidence_fields_exact=list(p.r.FIELDS))
 result=dict(status='PASS',scope='actual repair100.pipeline transform_layer on original single-layer50 checkpoint',
  frames=50,comparison_counts=comparison_counts,all_pixels_and_transaction_arrays_equal=True,
  aggregate_stage_results=reports,seconds=time.perf_counter()-started,
  limitations='This validates the layer adapter against frozen50 transactions and journals; it does not rerun model tracking or E1 region construction.')
 write('single_layer50_equivalence.json',result)
 return result

def check100():
 started=time.perf_counter()
 scene=p.r.Scene('room0')
 B=TOP/'prefix100'
 base=int(load(B/'association/observation_support_3cm.npz')['next_instance_id'][0])
 all_frames=[];layercounts={1:0,2:0};total_layers=0;rawcount=0
 for index,fid in enumerate(sorted(scene.records),1):
  frame=scene.source.load_frame(fid);layers=scene.identity_layers(fid,frame)
  assert len(layers)==(2 if fid<=245 else 1)
  layercounts[len(layers)]+=1;total_layers+=len(layers)
  parts=[]
  for image in layers:
   points,pids,u,v=p.h.sourcepixels(replace(frame,mask_local=image),scene)
   parts.append(p.h.keys(points,pids,base))
  actual=np.unique(np.concatenate(parts));cached=scene.support(fid)
  expected=p.h.keys(cached['surface_point_index'],cached['candidate_persistent_id'],base)
  equal(actual,expected,f'100 raw{fid} all-layer candidates')
  all_frames.append(strict_oracle(actual,base));rawcount+=len(actual)
  if index%20==0:stage_report('actual_pipeline_100_baseline',index,time.perf_counter()-started)
 k,v=aggregate(all_frames);ek,ev=pair_keys(scene.pair,base)
 equal(k,ek,'100 aggregate baseline keys');equal(v,ev,'100 aggregate baseline votes')
 evidence=p.r.reduce_surface_votes(k,v,len(scene.xyz),base,2,.67)
 for field in p.r.FIELDS:equal(evidence[field],scene.evidence[field],'100 baseline evidence '+field)
 scene.verify_unchanged()
 result=dict(status='PASS',scope='actual Scene100 identity_layers plus actual pipeline h.sourcepixels on all100 frames',
  frames=100,raw_cutoff=495,one_layer_frames=layercounts[1],two_layer_frames=layercounts[2],
  total_layers=total_layers,raw_candidate_keys_total=rawcount,unique_point_id_keys=len(k),effective_votes=int(v.sum()),
  all_frame_candidates_exact=True,strict_vote_aggregate_exact=True,evidence_exact=True,
  seconds=time.perf_counter()-started,repair100_executed=False)
 write('multilayer100_baseline.json',result)
 return result

def main():
 start=time.perf_counter()
 pipeline_sha=sha(HERE/'pipeline.py')
 write('layers_acceptance_input.json',dict(pipeline_sha256=pipeline_sha,validator_sha256=sha(__file__),started_unix=time.time()))
 original=read(TOP/'tracking100/A/input_freeze.json')['original_baseline_source_hashes']
 for path,digest in original.items():assert sha(path)==digest,path
 first=check50()
 second=check100()
 assert sha(HERE/'pipeline.py')==pipeline_sha,'pipeline changed during validation'
 for path,digest in original.items():assert sha(path)==digest,path
 result=dict(status='PASS',actual_pipeline_sha256=pipeline_sha,
  single_layer50_equivalence='single_layer50_equivalence.json',
  multilayer100_baseline='multilayer100_baseline.json',
  original_inputs_unchanged=True,repair_pipeline_main_called=False,map_write=False,
  seconds=time.perf_counter()-start,completed_unix=time.time())
 write('layers_acceptance.json',result)
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  write('layers_acceptance_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),unix=time.time()))
  raise
