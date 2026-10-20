"""Score frozen room0 maps with main's current R2 CLI; never alter predictions."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
import numpy as np
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019')
O=R/'comparison_three_way';MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
MANUAL=Path('/data/chenkejun/CVPR/results/room0_observation_repair_20261012')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,REGISTRY_PATH,require_current_protocol,require_current_gt
from unified_eval.io import load_gt,sha256_file as sha
from unified_eval.geometry import load_fixed_surface_correspondence
read=lambda p:json.loads(Path(p).read_text())

def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))

def cli(args,name):
 dump(O/'status.json',dict(status='RUNNING',stage=name,time=time.time()))
 with (O/(name+'.log')).open('x') as f:p=subprocess.run([sys.executable,'-m','unified_eval.cli',*map(str,args)],cwd=MAIN,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,name

def main():
 assert not O.exists();O.mkdir();verify_sources();require_current_protocol(CURRENT_CONFIG)
 original=read(R/'e1_repair/evaluation_r2/comparison.json');assert original['status']=='PASS'
 old_protocol=read(R/'e1_repair/evaluation_r2/protocol_freeze.json')
 assert sha(CURRENT_CONFIG)==old_protocol['current_protocol']['config_sha256']
 ref=read(MAIN/'instance_repair/frozen_e1_20261012/room0/reference.json')
 baseline=read(R/'e1_repair/original_repair/predictions_freeze.json')['scenes']['room0']['predictions']
 auto_root=R/'e1_repair/e1/E1_expanded_history'
 auto_freeze=read(auto_root/'predictions_freeze.json');manual_freeze=read(MANUAL/'E1_expanded_history/predictions_freeze.json')
 for f in [auto_freeze,manual_freeze]:assert f['status']=='PREDICTIONS_FROZEN_BEFORE_GT' and not f['GT_read_by_mapper']
 manual_objects=read(MANUAL/'objects.json');assert len(manual_objects)==41
 models=[dict(id='unrepaired',name='无修复（strict）',seed_count=0),dict(id='auto_v2',name='自动 v2 + E1',seed_count=27),dict(id='manual',name='人工标注种子 + E1',seed_count=41)]
 surfaces={};locks={}
 for stage in ['native','final']:
  for m in models:
   mid=m['id'];p=Path(baseline['strict_baseline_'+stage]['path']) if mid=='unrepaired' else (auto_root if mid=='auto_v2' else MANUAL/'E1_expanded_history')/'room0'/stage/'instance_surface.npz'
   expected=baseline['strict_baseline_'+stage]['sha256'] if mid=='unrepaired' else (auto_freeze if mid=='auto_v2' else manual_freeze)['output_sha256'][str(p)]
   assert sha(p)==expected
   if mid=='manual':assert expected==ref['reference_sha256'][str(p)]
   surfaces[(mid,stage)]=p;locks[str(p)]=expected
 dump(O/'inputs_frozen_before_scoring.json',dict(status='PASS',prediction_hashes=locks,manual_reference_sha256=sha(MAIN/'instance_repair/frozen_e1_20261012/room0/reference.json'),auto_prediction_freeze_sha256=sha(auto_root/'predictions_freeze.json'),manual_prediction_freeze_sha256=sha(MANUAL/'E1_expanded_history/predictions_freeze.json'),manual_policy=read(MANUAL/'policy.json'),models=models))
 gtpath=Path(CURRENT_PROTOCOL['canonical_GT']['room0']['file']);assert require_current_gt(gtpath)=='room0';gt=load_gt(gtpath)
 materialpath=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native/room0/P1-A1/surface_p0/materialization_report.json');associationpath=materialpath.parent.parent/'association/association_report.json'
 material=read(materialpath);association=read(associationpath);frames=[x['frame_id'] for x in material['frame_records']]
 assert frames==[x['frame_id'] for x in association['frame_summaries']]==gt.metadata['observation_metadata']['frame_ids']==list(range(0,2000,5))
 fsha=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest();assert fsha==gt.metadata['observation_metadata']['frame_list_sha256']
 config=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/configs/raw.json');assert sha(config)==material['config_sha256']==association['config_sha256']
 cache=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache/room0/fixed_gt_to_tsdf_r2.npz');assert sha(cache)==old_protocol['fixed_correspondence_sha256']
 rows=[];manifests=[];paired={};per_gt={};keys=('geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256')
 scope={int(x['raw_gt_id']):x for x in read(gtpath.parent/'gt_scope.json')['objects']}
 for stage in ['native','final']:
  reference=np.load(surfaces[('unrepaired',stage)]);overlaps={}
  for m in models:
   mid=m['id'];surface=surfaces[(mid,stage)];z=np.load(surface)
   for key in ['xyz_m','rgb']:np.testing.assert_array_equal(z[key],reference[key])
   load_fixed_surface_correspondence(cache,z['xyz_m'],gt.xyz_ref)
   provenance=dict(input_scope_verified=True,frame_list_sha256=fsha,frame_count=400,input_config_sha256=sha(config),materialization_report_sha256=sha(materialpath),association_report_sha256=sha(associationpath),prediction_frozen_before_GT=True,GT_used_for_mapping=False,condition=mid,selected_primary_masks=m['seed_count'],surface_sha256=sha(surface),inputs_freeze_sha256=sha(O/'inputs_frozen_before_scoring.json'))
   pp=O/'provenance'/(mid+'_'+stage+'.json');dump(pp,provenance);dest=O/(mid+'_'+stage)
   cli(['adapt-surface','--gt',gtpath,'--surface',surface,'--fixed-correspondence',cache,'--source-provenance',pp,'--method-name','room0_three_way_'+mid+'_'+stage,'--method-commit',sha(C/'compare_three_way.py'),'--out',dest],mid+'_'+stage)
   sm=read(dest/'metrics.json')['summary'];am=read(dest/'adapter_manifest.json');manifests.append(am)
   row=dict(**m,stage=stage,F1=sm['CA_PRF1_0_5']['F1'],PQ=sm['CA_PQ']['PQ'],AP=sm['CA_AP_uniform'],AP50=sm['CA_AP50_uniform'],mCov=sm['CA_mCov'],TP=sm['CA_PRF1_0_5']['TP'],FP=sm['CA_PRF1_0_5']['FP'],FN=sm['CA_PRF1_0_5']['FN'],gray_total=int((z['instance_id']<=0).sum()),owner_surface=sm['owner_surface'])
   previous=ref['R2_final'] if mid=='manual' and stage=='final' else next((r for r in original['rows'] if r['stage']==stage and r['variant']==('before' if mid=='unrepaired' else 'after')),None) if mid!='manual' else None
   if previous:
    for key in ['F1','PQ','AP','AP50','mCov','TP','FP','FN','gray_total']:assert abs(row[key]-previous[key])<1e-12,(mid,stage,key)
   rows.append(row);print('SCORED',json.dumps(row,ensure_ascii=False),flush=True)
   overlaps[mid]=np.load(dest/'overlap_matrix.npz')
  paired[stage]={}
  for before,after in [('unrepaired','auto_v2'),('unrepaired','manual'),('manual','auto_v2')]:
   name=before+'_to_'+after;dest=O/'paired'/stage/name
   cli(['eval-repair-pair','--gt',gtpath,'--before',O/(before+'_'+stage)/'canonical_prediction.npz','--after',O/(after+'_'+stage)/'canonical_prediction.npz','--out',dest],'paired_'+stage+'_'+name)
   paired[stage][name]=read(dest/'paired_repair_metrics.json')
  for z in overlaps.values():np.testing.assert_array_equal(z['gt_ids'],overlaps['unrepaired']['gt_ids'])
  per_gt[stage]=[dict(GT_ID=int(gid),semantic_class=scope[int(gid)]['semantic_class'],**{mid:float(z['iou'][:,i].max(initial=0)) for mid,z in overlaps.items()}) for i,gid in enumerate(overlaps['unrepaired']['gt_ids'])]
  assert len(per_gt[stage])==72
 for m in manifests[1:]:
  for k in keys:assert m[k]==manifests[0][k],k
 for p,h in locks.items():assert sha(p)==h,p
 verify_sources()
 consistency=dict(status='PASS',same_geometry_RGB=True,same_GT_scope_and_observed_support=True,same_fixed_correspondence=True,same_R2_config=True,predictions_unchanged=True,manual_matches_frozen_main_reference=True,auto_and_unrepaired_match_previous_scores=True,GT_count=72,frame_count=400,current_protocol=CURRENT_PROTOCOL,registry_sha256=sha(REGISTRY_PATH),fixed_correspondence_sha256=sha(cache),shared_manifest_fields={k:manifests[0][k] for k in keys},main_entry_commit=subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip())
 dump(O/'consistency.json',consistency)
 result=dict(status='PASS',scene='room0',protocol='object_observed_repair / revision 2',frozen=False,GT_count=72,models=models,rows=rows,paired=paired,per_GT=per_gt,manual_source=str(MANUAL),auto_source=str(auto_root),unrepaired_source=str(surfaces[('unrepaired','final')]),comparison_scope='Same fixed TSDF and R2 GT; seed sets and their selected identities differ. Manual means 41 human-selected seed masks followed by original E1, not GT-as-prediction.',M0466_CropFormer_correction_preserved=True)
 dump(O/'summary.json',result);dump(O/'status.json',dict(status='PASS',stage='complete',time=time.time()));print('THREE_WAY_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  dump(O/'failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
