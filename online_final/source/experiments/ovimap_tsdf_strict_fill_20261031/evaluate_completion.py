from common import *
import subprocess
sys.path.insert(0,str(OLD_C))
import official_rule_eval as rule
from evaluate_bounded import fields
from plyfile import PlyData
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONPATH':str(MAIN)}
def run(cmd,log):
 with Path(log).open('w') as f:subprocess.run([str(x) for x in cmd],cwd=MAIN,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
def evaluate(scene):
 dst=O/'scenes'/scene;mapping=read(dst/'mapping_complete.json');assert mapping['status']=='PASS'
 predpath=dst/'instance_surface.npz';assert sha(predpath)==mapping['prediction_sha256']
 pred=load(predpath);native_path=OLD/'native'/f'{scene}_400_trace/ovi/f001995.npz'
 native=load(native_path);np.testing.assert_array_equal(pred['xyz_m'],native['xyz_m'])
 old_primary=read(OLD/'official_rule_evaluation'/scene/'complete.json')
 assert sha(native_path)==old_primary['prediction_sha256']['ovimap']
 gt=Path(old_primary['GT_file']);assert sha(gt)==old_primary['GT_file_sha256']
 vertices=PlyData.read(gt)['vertex'].data;ref=np.column_stack([vertices[k] for k in ['x','y','z']]).astype(np.float32)
 own,ovi,cor=rule.project_pair(pred['xyz_m'],ref,np.maximum(pred['instance_id'],0),np.maximum(native['instance_id'],0))
 old_cor=load(OLD/'official_rule_evaluation'/scene/'shared_correspondence.npz')
 for k in cor:np.testing.assert_array_equal(cor[k],old_cor[k])
 old_ovi=PlyData.read(OLD/'official_rule_evaluation'/scene/'ovimap_on_gt.ply')['vertex'].data['label']
 np.testing.assert_array_equal(ovi,old_ovi)
 primary=dst/'primary';primary.mkdir(exist_ok=True)
 outfile=primary/'filled_on_gt.ply';rule.write_labels(outfile,ref,own)
 metrics,stdout=rule.score(gt,outfile);(primary/'upstream_stdout.txt').write_text(stdout)
 assert metrics['GT_count']==old_primary['GT_instances_min10']
 dump(primary/'comparison.json',dict(ours_filled=metrics,ours_unfilled=old_primary['comparison']['ours'],ovimap=old_primary['comparison']['ovimap']))
 # Current main R2 remains the auxiliary evaluator, with the already certified shared correspondence.
 reference=read(OLD/'evaluation'/scene/'ours/adapter_manifest.json')
 lock=read(MAIN/'unified_eval/configs/current_protocol.json')
 assert reference['profile_config_sha256']==lock['config_sha256']
 prov={**reference['source_provenance'],'experiment':'ovimap_tsdf_strict_fill_20261031',
  'source_map_sha256':sha(predpath),'publication':'strict >=2 and >=0.67 plus frozen holes_geodesic',
  'repair_enabled':False,'hole_completion_enabled':True,'GT_used_for_mapping':False,
  'completion_report':str(dst/'mapping_complete.json'),'completion_report_sha256':sha(dst/'mapping_complete.json'),
  'completion_feedback_to_votes':False,'stage':'final_publication_only'}
 provfile=dst/'filled_provenance.json';dump(provfile,prov)
 cache=Path(reference['correspondence_cache']);cachehash=sha(cache)
 run([P,'-B','-m','unified_eval.cli','adapt-surface','--gt',reference['gt_file'],'--surface',predpath,
  '--fixed-correspondence',cache,'--source-provenance',provfile,'--method-name','ours_strict_plus_holes_geodesic',
  '--method-commit',sha(C/'fill_surface.py'),'--out',dst/'r2'],dst/'r2.log')
 manifest=read(dst/'r2/adapter_manifest.json')
 for k in ['gt_file_sha256','profile_config_sha256','evaluator_code_sha256','correspondence_file_sha256']:
  assert manifest[k]==reference[k],k
 assert sha(cache)==cachehash and sha(predpath)==mapping['prediction_sha256']
 aux=fields(read(dst/'r2/metrics.json')['summary'])
 old_aux=read(OLD/'evaluation'/scene/'comparison.json')
 report=dict(status='PASS',scene=scene,primary=read(primary/'comparison.json'),
  auxiliary=dict(ours_filled=aux,ours_unfilled=old_aux['ours'],ovimap=old_aux['ovimap']),
  primary_same_correspondence=True,R2_same_correspondence=True,
  prediction_sha256=mapping['prediction_sha256'],GT_not_used_for_completion=True,
  filled_points=mapping['filled_points'],existing_IDs_changed=0,new_instance_IDs=0)
 dump(dst/'complete.json',report);print('EVALUATED',scene,json.dumps(report),flush=True)
def aggregate():
 reports={s:read(O/'scenes'/s/'complete.json') for s in SCENES};assert all(x['status']=='PASS' for x in reports.values())
 metrics=['mIoU','wIoU','P@25','R@25','P@50','R@50','P@75','R@75']
 primary={m:{k:float(np.mean([reports[s]['primary'][m][k] for s in SCENES])) for k in metrics}
          for m in ['ours_unfilled','ours_filled','ovimap']}
 rows=[]
 for s in SCENES:
  path=O/'scenes'/s/'r2';m=read(path/'adapter_manifest.json')
  rows.append(dict(gt=m['gt_file'],prediction=str(path/'canonical_prediction.npz'),diagnostic_prediction=str(path/'diagnostic_support_prediction.npz')))
 pool=O/'pooled';pool.mkdir(exist_ok=True);dump(pool/'batch.json',dict(scenes=rows))
 run([P,'-B','-m','unified_eval.cli','eval-batch','--scenes',pool/'batch.json','--out',pool],pool/'eval.log')
 aux=read(OLD/'comparison.json')
 auxiliary=dict(ours_unfilled=aux['ours'],ours_filled=fields(read(pool/'summary.json')),ovimap=aux['ovimap'])
 assert auxiliary['ours_filled']['TP']+auxiliary['ours_filled']['FN']==350
 out=dict(status='PASS',scenes=8,frames_per_scene=400,primary_GT=542,auxiliary_GT=350,
  primary_aggregation='unweighted eight-scene mean',primary=primary,auxiliary=auxiliary,
  filled_points_total=sum(r['filled_points'] for r in reports.values()),
  definition='original strict one-vote plus original holes_geodesic final publication; no repair feedback',
  existing_IDs_changed=0,new_instance_IDs=0,geometry_changed=False)
 dump(O/'comparison.json',out);print('COMPARISON',json.dumps(out),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='aggregate':aggregate()
 else:evaluate(sys.argv[1])
