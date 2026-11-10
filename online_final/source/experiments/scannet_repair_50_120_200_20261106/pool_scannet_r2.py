from pathlib import Path
import sys,json,hashlib,csv
C=Path('/home/chenkejun/CVPR/experiments/scannet_repair_50_120_200_20261106')
O=Path('/data/chenkejun/CVPR/results/scannet_repair_50_120_200_20261106')
OLD=Path('/data/chenkejun/CVPR/results/scannet_ovimap_strict_dbscan_20261104')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');sys.path.insert(0,str(MAIN))
from unified_eval.cli import build_parser,cmd_eval_batch
from unified_eval.current_protocol import require_current_protocol,reject_current_debug_overrides,CURRENT_CONFIG
from unified_eval.io import repair_evaluator_code_hashes
variant=sys.argv[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
scope=read(OLD/'ground_truth/scope_lock.json');assert scope['status']=='PASS' and scope['frame_count']==200
assert scope['evaluator_code_sha256']==repair_evaluator_code_hashes() and scope['current_config_sha256']==sha(CURRENT_CONFIG)
protocol=read(MAIN/'unified_eval/configs/current_protocol.json');assert protocol['profile_revision']==2 and not protocol['frozen']
out=O/'aggregate'/variant;out.mkdir(parents=True,exist_ok=True);rows=[];proof=[]
for scene in ['scene0011_00','scene0050_00']:
 gt=OLD/'ground_truth'/scene/'ground_truth/gt.npz'
 if variant in ('no_repair','OVIMAP'):rd=OLD/'evaluation'/variant/scene/'r2'
 elif variant in ('main','online_B'):rd=O/'evaluation'/variant/scene/'r2'
 elif variant in ('main_DBSCAN','online_B_DBSCAN'):rd=O/'dbscan'/variant.replace('_DBSCAN','')/scene/'r2'
 else:raise ValueError(variant)
 adapter=read(rd/'adapter_manifest.json');baseline=read(OLD/'evaluation/no_repair'/scene/'r2/adapter_manifest.json')
 assert adapter['gt_file_sha256']==sha(gt) and adapter['gt_scope_sha256']==scope['scope'][scene]['gt_scope_sha256'] and adapter['profile_config_sha256']==sha(CURRENT_CONFIG)
 if variant!='OVIMAP':
  for k in ['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256']:assert adapter[k]==baseline[k],(scene,k)
 pred=rd/'canonical_prediction.npz';diag=rd/'diagnostic_support_prediction.npz'
 assert sha(pred)==adapter['canonical_prediction_sha256'] and sha(diag)==adapter['diagnostic_prediction_sha256']
 rows.append({'gt':str(gt),'prediction':str(pred),'diagnostic_prediction':str(diag)})
 proof.append({'scene':scene,'gt_sha256':sha(gt),'prediction_sha256':sha(pred),'adapter_sha256':sha(rd/'adapter_manifest.json')})
manifest=out/'scenes.json';manifest.write_text(json.dumps({'scenes':rows},indent=2))
args=build_parser().parse_args(['eval-batch','--scenes',str(manifest),'--out',str(out)])
require_current_protocol(args.config,historical_protocol=False);reject_current_debug_overrides(args);cmd_eval_batch(args)
with (out/'per_scene.csv').open() as f:result=list(csv.DictReader(f))
for row,entry in zip(result,rows):
 m=read(Path(entry['prediction']).parent/'metrics.json')['summary']
 assert abs(float(row['CA_AP50_uniform'])-m['CA_AP50_uniform'])<1e-12
 assert int(row['TP'])==m['CA_PQ']['TP'] and int(row['FN'])==m['CA_PQ']['FN']
summary=read(out/'summary.json');assert summary['CA_PRF1_0_5']['TP']+summary['CA_PRF1_0_5']['FN']==52
(out/'scope_audit.json').write_text(json.dumps({'status':'PASS','GT':52,'scope_lock_sha256':sha(OLD/'ground_truth/scope_lock.json'),'unchanged_main_batch':True,'per_scene_reproduces_single_results':True,'fixed_geometry_comparison':variant!='OVIMAP','inputs':proof},indent=2))
print(json.dumps({'status':'PASS','variant':variant,'F1':summary['CA_PRF1_0_5']['F1'],'PQ':summary['CA_PQ']['PQ'],'AP':summary['CA_AP_uniform'],'AP50':summary['CA_AP50_uniform'],'mCov':summary['CA_mCov']}))
