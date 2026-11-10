"""Score frozen maps only, through main's current default revision-2 CLI."""
from pathlib import Path
import json,hashlib,subprocess,sys,csv,importlib.util,gc
import numpy as np
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/mask_gate_no_foreign_v2_20261010/room0')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
OUT=R/'evaluation_r2';sys.path.insert(0,str(MAIN))
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,REGISTRY_PATH,require_current_protocol,require_current_gt
from unified_eval.io import load_gt,load_prediction,sha256_file
from unified_eval.geometry import load_fixed_surface_correspondence
from unified_eval.schema import Protocol
spec=importlib.util.spec_from_file_location('verified_r2_analysis', '/home/chenkejun/CVPR/experiments/p1a1_history_core_reassociation_20261009/evaluate_history_core_r2.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cli(args):
 p=subprocess.run([sys.executable,'-m','unified_eval.cli',*map(str,args)],cwd=MAIN,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 print(p.stdout.strip(),flush=True)
def main():
 assert not OUT.exists()
 a=read(R/'original_repair/predictions_freeze.json');b=read(R/'modified_repair/predictions_freeze.json')
 assert not a['GT_used_for_mapping'] and not b['GT_read_by_mapper'] and not b['baseline_modified']
 for freeze,groups in [(a,['generated_output_sha256','new_code_sha256']),(b,['output_sha256','code_sha256'])]:
  for k in groups:
   for p,h in freeze[k].items():assert sha256_file(p)==h,p
 assert require_current_protocol(CURRENT_CONFIG)
 assert CURRENT_PROTOCOL['profile_revision']==2 and CURRENT_PROTOCOL['evaluable_GT_count']==350 and not CURRENT_PROTOCOL['frozen']
 assert CURRENT_PROTOCOL['scoring_commit']=='cd260569e89de1e9525f82c45ee10842b80616b0'
 for name,h in CURRENT_PROTOCOL['evaluator_code_sha256'].items():assert sha256_file(MAIN/'unified_eval'/name)==h
 protocol=Protocol.from_dict(read(CURRENT_CONFIG));gtpath=Path(CURRENT_PROTOCOL['canonical_GT']['room0']['file']);assert require_current_gt(gtpath)=='room0'
 dump(OUT/'protocol_freeze.json',{'current_protocol':CURRENT_PROTOCOL,'registry_sha256':sha256_file(REGISTRY_PATH),'entry_commit':subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip(),'predictions_frozen_before_GT':True,'default_main_CLI':True})
 gt=load_gt(gtpath);scope_objects={r['raw_gt_id']:r for r in read(gtpath.parent/'gt_scope.json')['objects']}
 records=a['scenes']['room0']['predictions']
 cache=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache/room0/fixed_gt_to_tsdf_r2.npz');cache_sha=sha256_file(cache)
 with np.load(records['strict_baseline_final']['path']) as z:xyz=z['xyz_m'].copy()
 correspondence=load_fixed_surface_correspondence(cache,xyz,gt.xyz_ref)
 materialpath=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native/room0/P1-A1/surface_p0/materialization_report.json');material=read(materialpath)
 frames=[v['frame_id'] for v in material['frame_records']];assert frames==gt.metadata['observation_metadata']['frame_ids'] and frames==list(range(0,2000,5))
 framesha=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest();assert framesha==gt.metadata['observation_metadata']['frame_list_sha256']
 # Define selected GT only after predictions are frozen, using each human seed's dominant evaluable reference GT.
 seedtruth=[];targets=set();nearest=correspondence.nearest_native_index;exists=nearest>=0
 eval_ids={k for k,v in scope_objects.items() if v['evaluable']}
 for case in read(R/'seed_diagnostics.json')['cases']:
  with np.load(R/'cases'/case['case_uid']/'seed_projected_support.npz') as z:
   for obj in case['objects']:
    pts=z['surface_point_index'][z['track_id']==obj['track_id']];ref=exists&np.isin(nearest,pts)&np.isin(gt.instance_id,list(eval_ids))
    ids,counts=np.unique(gt.instance_id[ref],return_counts=True);dominant=int(ids[counts.argmax()]) if len(ids) else None
    if dominant is not None:targets.add(dominant)
    seedtruth.append({'ROI':case['source_choice']['ROI'],'track_id':obj['track_id'],'mask_id':obj['native_mask_id'],'dominant_evaluable_GT':dominant,'GT_histogram':{str(int(k)):int(v) for k,v in zip(ids,counts)}})
 dump(OUT/'selected_targets_posthoc.json',{'definition':'Dominant evaluable GT of each frozen manual seed under fixed reference correspondence; posthoc analysis only','GT_not_used_for_mapping':True,'targets':sorted(targets),'seeds':seedtruth})
 scores={};states={};pairs={};rows=[]
 for stage in ['native','final']:
  maps={'baseline':Path(records['strict_baseline_'+stage]['path']),'original_repair':Path(records['strict_full_track_'+stage]['path']),'modified_repair':R/'modified_repair/room0'/stage/'instance_surface.npz'}
  qualities={};manifests={};destinations={};categories={}
  for condition,surface in maps.items():
   name=condition+'_'+stage;dest=OUT/'evaluation'/name;destinations[condition]=dest
   with np.load(surface) as z:np.testing.assert_array_equal(xyz,z['xyz_m']);labels=z['instance_id'].copy()
   evidencepath=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room0/abstain/surface_evidence.npz') if condition=='baseline' else (R/'original_repair/room0/full_track/native/surface_evidence.npz' if condition=='original_repair' else R/'modified_repair/room0/native/surface_evidence.npz')
   with np.load(evidencepath) as z:state=z['state'].copy()
   category=np.where(labels>0,2,state);assert not np.any((labels<=0)&(state==2));categories[condition]=category
   states[name]={'U':int(np.sum(category==0)),'T':int(np.sum(category==1)),'C':int(np.sum(category==3)),'gray_total':int(np.sum(labels<=0)),'native_state_counts_U_T_confirmed_C':np.bincount(state,minlength=4).tolist()}
   prov=OUT/'provenance'/(name+'.json')
   dump(prov,{'input_scope_verified':True,'frame_count':400,'frame_list_sha256':framesha,'input_config_sha256':material['config_sha256'],'materialization_report_sha256':sha256_file(materialpath),'GT_read_by_mapper':False,'prediction_freeze_sha256':sha256_file(R/'modified_repair/predictions_freeze.json'),'fixed_R2_correspondence_reused':str(cache)})
   cli(['adapt-surface','--gt',gtpath,'--surface',surface,'--fixed-correspondence',cache,'--source-provenance',prov,'--method-name','P1-A1_strict_room0_'+name,'--method-commit',sha256_file(C/'run_history_core.py'),'--out',dest])
   am=read(dest/'adapter_manifest.json');assert am['profile_revision']==2 and am['profile_config_sha256']==CURRENT_PROTOCOL['config_sha256'] and am['correspondence_file_sha256']==cache_sha;manifests[condition]=am
   pred=load_prediction(dest/'canonical_prediction.npz');qualities[condition]=helper.quality(gt,pred,protocol,labels,correspondence)
   scores[name]=helper.fields(read(dest/'metrics.json')['summary']);rows.append({'condition':name,**scores[name],**{k:v for k,v in states[name].items() if k!='native_state_counts_U_T_confirmed_C'}})
   print('SCORED',name,json.dumps(scores[name]),flush=True)
  for before,after in [('baseline','original_repair'),('baseline','modified_repair'),('original_repair','modified_repair')]:
   for key in ['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256']:assert manifests[before][key]==manifests[after][key],key
   name=after+'_vs_'+before+'_'+stage;dest=OUT/'paired'/name
   cli(['eval-repair-pair','--gt',gtpath,'--before',destinations[before]/'canonical_prediction.npz','--after',destinations[after]/'canonical_prediction.npz','--out',dest])
   pair=helper.enrich_pair(read(dest/'paired_repair_metrics.json'),qualities[before],qualities[after],targets,'room0',scope_objects)
   pair['native_point_state_transition_U_T_assigned_C']=np.bincount(categories[before].astype(np.int64)*4+categories[after],minlength=16).reshape(4,4).tolist()
   dump(dest/'analysis.json',pair);pairs[name]=pair
  del qualities;gc.collect()
 dump(OUT/'comparison.json',{'status':'PASS','scene':'room0','evaluation_profile':'object_observed_repair','profile_revision':2,'fixed_scene_GT':72,'frozen':False,'same_geometry_scope_correspondence':True,'scores':scores,'states':states,'paired':pairs,'selected_target_GT_ids_posthoc':sorted(targets),'human_annotation_sha256':sha256_file(R/'inputs/human_selection.json')})
 with (OUT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('EVALUATION_PASS',flush=True)
if __name__=='__main__':main()
