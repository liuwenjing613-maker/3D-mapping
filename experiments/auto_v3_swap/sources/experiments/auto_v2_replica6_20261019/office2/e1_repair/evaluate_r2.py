"""Evaluate only frozen predictions using main's current R2 default CLI."""
from pathlib import Path
import json,sys,hashlib,subprocess,time
import numpy as np
C=Path(__file__).resolve().parent;R=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office2/e1_repair');MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,REGISTRY_PATH,require_current_protocol,require_current_gt
from unified_eval.io import load_gt,sha256_file
from unified_eval.geometry import load_fixed_surface_correspondence,build_fixed_surface_correspondence,save_fixed_surface_correspondence
read=lambda p:json.loads(Path(p).read_text())
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def cli(args,log):
 with log.open('w') as f:p=subprocess.run([sys.executable,'-m','unified_eval.cli',*map(str,args)],cwd=MAIN,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,str(log)
def main():
 verify_sources();assert read(R/'pipeline_complete.json')['status']=='PASS'
 finalroot=R/'e1/E1_expanded_history';freeze=read(finalroot/'predictions_freeze.json')
 assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT' and not freeze['GT_read_by_mapper']
 for kind in ['output_sha256','code_sha256']:
  for p,h in freeze[kind].items():assert sha256_file(p)==h,p
 assert require_current_protocol(CURRENT_CONFIG)
 out=R/'evaluation_r2';assert not (out/'comparison.json').exists();out.mkdir(exist_ok=True)
 gtpath=Path(CURRENT_PROTOCOL['canonical_GT']['office2']['file']);assert require_current_gt(gtpath)=='office2';gt=load_gt(gtpath)
 materialpath=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native/office2/P1-A1/surface_p0/materialization_report.json')
 associationpath=materialpath.parent.parent/'association/association_report.json'
 material=read(materialpath);association=read(associationpath);frames=[x['frame_id'] for x in material['frame_records']]
 assert frames==[x['frame_id'] for x in association['frame_summaries']]==gt.metadata['observation_metadata']['frame_ids']==list(range(0,2000,5))
 fsha=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest();assert fsha==gt.metadata['observation_metadata']['frame_list_sha256']
 config=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/office2/configs/raw.json');assert sha256_file(config)==material['config_sha256']==association['config_sha256']
 cache=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/geometry_cache/office2/fixed_gt_to_tsdf_r2.npz')
 baseline_for_cache=read(R/'original_repair/predictions_freeze.json')['scenes']['office2']['predictions']['strict_baseline_final']['path']
 if not cache.exists():
  cache.parent.mkdir(parents=True,exist_ok=True)
  with np.load(baseline_for_cache) as zz:correspondence=build_fixed_surface_correspondence(zz['xyz_m'],gt.xyz_ref,read(CURRENT_CONFIG)['geometry_mapping']['max_distance_m'])
  save_fixed_surface_correspondence(cache,correspondence)
 protocol={'current_protocol':CURRENT_PROTOCOL,'registry_sha256':sha256_file(REGISTRY_PATH),'default_main_CLI':True,'entry_commit':subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip(),'predictions_frozen_before_GT':True,'fixed_correspondence_sha256':sha256_file(cache)}
 dump(out/'protocol_freeze.json',protocol)
 baseline=read(R/'original_repair/predictions_freeze.json')['scenes']['office2']['predictions'];rows=[];manifests=[];allstats={}
 for stage in ('native','final'):
  for variant in ('before','after'):
   surface=Path(baseline['strict_baseline_'+stage]['path']) if variant=='before' else finalroot/'office2'/stage/'instance_surface.npz'
   z=np.load(surface);reference=np.load(baseline['strict_baseline_'+stage]['path']);np.testing.assert_array_equal(z['xyz_m'],reference['xyz_m']);np.testing.assert_array_equal(z['rgb'],reference['rgb'])
   load_fixed_surface_correspondence(cache,z['xyz_m'],gt.xyz_ref)
   provenance={'input_scope_verified':True,'frame_list_sha256':fsha,'frame_count':400,'input_config_sha256':sha256_file(config),'materialization_report_sha256':sha256_file(materialpath),'association_report_sha256':sha256_file(associationpath),
    'prediction_freeze_sha256':sha256_file(finalroot/'predictions_freeze.json'),'prediction_frozen_before_GT':True,'GT_used_for_mapping':False,'condition':variant,'selected_primary_masks':len(read(R/'seed_id_map.json')['objects']),'batch':str(R),'surface_sha256':sha256_file(surface)}
   pp=out/'provenance'/(variant+'_'+stage+'.json');dump(pp,provenance);dest=out/(variant+'_'+stage)
   cli(['adapt-surface','--gt',gtpath,'--surface',surface,'--fixed-correspondence',cache,'--source-provenance',pp,'--method-name','P1-A1_strict_office2_autov2_E1_'+variant+'_'+stage,'--method-commit',sha256_file(C/'run_observation_repair.py'),'--out',dest],out/(variant+'_'+stage+'.log'))
   am=read(dest/'adapter_manifest.json');manifests.append(am)
   sm=read(dest/'metrics.json')['summary'];labels=z['instance_id'];e=np.load('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/baseline/ablation/office2/abstain/surface_evidence.npz') if variant=='before' else np.load(finalroot/'office2/native/surface_evidence.npz')
   counts={n:int(np.sum((labels<=0)&(e['state']==v))) for n,v in [('U',0),('T',1),('C',3)]}
   row={'variant':variant,'stage':stage,'F1':sm['CA_PRF1_0_5']['F1'],'PQ':sm['CA_PQ']['PQ'],'AP':sm['CA_AP_uniform'],'AP50':sm['CA_AP50_uniform'],'mCov':sm['CA_mCov'],'TP':sm['CA_PRF1_0_5']['TP'],'FP':sm['CA_PRF1_0_5']['FP'],'FN':sm['CA_PRF1_0_5']['FN'],'gray_total':int((labels<=0).sum()),'gray_states':counts,'owner_surface':sm['owner_surface']}
   rows.append(row);allstats[variant+'_'+stage]=sm;print('SCORED',json.dumps(row),flush=True)
  pair=out/'paired'/stage
  cli(['eval-repair-pair','--gt',gtpath,'--before',out/('before_'+stage)/'canonical_prediction.npz','--after',out/('after_'+stage)/'canonical_prediction.npz','--out',pair],out/('paired_'+stage+'.log'))
 for m in manifests[1:]:
  for key in ('geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256'):assert m[key]==manifests[0][key],key
 pergt=[];scope={int(x['raw_gt_id']):x for x in read(gtpath.parent/'gt_scope.json')['objects']}
 before=np.load(out/'before_final/overlap_matrix.npz');after=np.load(out/'after_final/overlap_matrix.npz');assert set(before['gt_ids'])==set(after['gt_ids'])
 for gid in before['gt_ids']:
  i=int(np.flatnonzero(before['gt_ids']==gid)[0]);j=int(np.flatnonzero(after['gt_ids']==gid)[0]);a=float(before['iou'][:,i].max(initial=0));b=float(after['iou'][:,j].max(initial=0))
  pergt.append({'GT_ID':int(gid),'semantic_class':scope[int(gid)]['semantic_class'],'before_best_iou':a,'after_best_iou':b,'delta':b-a})
 dump(out/'per_GT.json',pergt)
 result={'status':'PASS','protocol':'object_observed_repair / revision 2','development_protocol_frozen':False,'scene':'office2','GT_count':len(pergt),'control':'same strict pre-repair map used by ROI selection','rows':rows,
  'GT_best_IoU_changes':{name:sum(f(x['delta']) for x in pergt) for name,f in [('improved',lambda x:x>1e-12),('worsened',lambda x:x< -1e-12),('unchanged',lambda x:abs(x)<=1e-12)]},
  'same_geometry_GT_scope_and_correspondence':True,'predictions_frozen_before_GT':True}
 dump(out/'comparison.json',result);dump(R/'evaluation_complete.json',result)
 verify_sources();print('EVALUATION_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
