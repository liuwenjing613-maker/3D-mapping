"""Evaluate C first; B scoring deliberately deferred until C has been delivered."""
from pathlib import Path
import sys,json,subprocess,importlib.util,hashlib
import numpy as np
C=Path(__file__).resolve().parent;SCENE=C.name
R=Path('/data/chenkejun/CVPR/results/mask_gate_no_foreign_v2_20261010')/SCENE
BATCH={'room0':'batch_918663ef2e2e','room2':'batch_20fd547a5808'}[SCENE]
ORIGINAL=Path('/data/chenkejun/CVPR/results')/(SCENE+'_strict_manual_ab_20261010')/BATCH
OLD=Path('/data/chenkejun/CVPR/results/anchor_guided_gate_v1_20261010')/SCENE
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');sys.path.insert(0,str(MAIN))
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,REGISTRY_PATH,require_current_protocol,require_current_gt
from unified_eval.io import load_gt,load_prediction,sha256_file
from unified_eval.geometry import load_fixed_surface_correspondence
from unified_eval.schema import Protocol
spec=importlib.util.spec_from_file_location('helper','/home/chenkejun/CVPR/experiments/p1a1_history_core_reassociation_20261009/evaluate_history_core_r2.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def cli(args):
 p=subprocess.run([sys.executable,'-m','unified_eval.cli',*map(str,args)],cwd=MAIN,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
def main():
 out=R/'evaluation_c_r2';assert not out.exists()
 freeze=read(R/'modified_repair/predictions_freeze.json');assert not freeze['GT_read_by_mapper']
 for group in ['output_sha256','code_sha256']:
  for p,h in freeze[group].items():assert sha256_file(p)==h,p
 assert require_current_protocol(CURRENT_CONFIG)
 assert CURRENT_PROTOCOL['config_sha256']=='6668792cbbeff3b6a6b6f2302ad8baa1b202f463ae26306350d4617c07f601a0'
 for p,h in CURRENT_PROTOCOL['evaluator_code_sha256'].items():assert sha256_file(MAIN/'unified_eval'/p)==h
 gtpath=Path(CURRENT_PROTOCOL['canonical_GT'][SCENE]['file']);assert require_current_gt(gtpath)==SCENE
 gt=load_gt(gtpath);protocol=Protocol.from_dict(read(CURRENT_CONFIG))
 scope={r['raw_gt_id']:r for r in read(gtpath.parent/'gt_scope.json')['objects']}
 targets=read(ORIGINAL/'evaluation_r2/selected_targets_posthoc.json')['targets']
 cache=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache')/SCENE/'fixed_gt_to_tsdf_r2.npz'
 dump(out/'protocol_freeze.json',{'current_protocol':CURRENT_PROTOCOL,'registry_sha256':sha256_file(REGISTRY_PATH),'entry_commit':subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip(),'default_main_CLI':True,'predictions_frozen_before_GT':True})
 rows=[];pairs={}
 for stage in ['native','final']:
  surface=R/'modified_repair'/SCENE/stage/'instance_surface.npz';oldsurface=OLD/'modified_repair'/SCENE/stage/'instance_surface.npz'
  with np.load(surface) as z:xyz=z['xyz_m'].copy();labels=z['instance_id'].copy()
  with np.load(oldsurface) as z:np.testing.assert_array_equal(xyz,z['xyz_m']);oldlabels=z['instance_id'].copy()
  corr=load_fixed_surface_correspondence(cache,xyz,gt.xyz_ref)
  prov=read(OLD/'evaluation_c_r2/provenance'/(stage+'.json'));prov['prediction_freeze_sha256']=sha256_file(R/'modified_repair/predictions_freeze.json')
  prov['gate_policy_sha256']=sha256_file(C/'gate_policy.json');pp=out/'provenance'/(stage+'.json');dump(pp,prov)
  dest=out/('modified_repair_'+stage);oldest=OLD/'evaluation_c_r2'/('modified_repair_'+stage)
  cli(['adapt-surface','--gt',gtpath,'--surface',surface,'--fixed-correspondence',cache,'--source-provenance',pp,'--method-name','P1-A1_strict_C_MaskGateV2NoForeign_'+SCENE+'_'+stage,'--method-commit',sha256_file(C/'run_batch.py'),'--out',dest])
  am=read(dest/'adapter_manifest.json');ref=read(oldest/'adapter_manifest.json')
  for k in ['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256']:assert am[k]==ref[k],k
  with np.load(R/'modified_repair'/SCENE/'native/surface_evidence.npz') as z:cat=np.where(labels>0,2,z['state'])
  with np.load(OLD/'modified_repair'/SCENE/'native/surface_evidence.npz') as z:oldcat=np.where(oldlabels>0,2,z['state'])
  assert not np.any((labels<=0)&(cat==2))
  row={'scene':SCENE,'stage':stage,**helper.fields(read(dest/'metrics.json')['summary']),**{k:int(np.sum(cat==v)) for k,v in [('U',0),('T',1),('C',3)]},'gray_total':int(np.sum(labels<=0))};rows.append(row)
  qualities=[helper.quality(gt,load_prediction(p/'canonical_prediction.npz'),protocol,l,corr) for p,l in [(oldest,oldlabels),(dest,labels)]]
  paired=out/'paired'/stage;cli(['eval-repair-pair','--gt',gtpath,'--before',oldest/'canonical_prediction.npz','--after',dest/'canonical_prediction.npz','--out',paired])
  pair=helper.enrich_pair(read(paired/'paired_repair_metrics.json'),*qualities,targets,SCENE,scope)
  pair['native_point_state_transition_U_T_assigned_C']=np.bincount(oldcat.astype(np.int64)*4+cat,minlength=16).reshape(4,4).tolist();pairs[stage]=pair;dump(paired/'analysis.json',pair)
  print('C_SCORED',SCENE,stage,json.dumps(row),flush=True)
 dump(out/'comparison.json',{'status':'PASS','rows':rows,'paired':pairs,'comparison':'new C gate V2 versus previous mask_gate C; B scoring deferred'})
if __name__=='__main__':main()
