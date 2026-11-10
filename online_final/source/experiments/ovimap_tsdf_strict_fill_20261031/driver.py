from common import *
import subprocess,concurrent.futures,traceback
from native_runner import container
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENCV_FOR_THREADS_NUM':'1','PYTHONDONTWRITEBYTECODE':'1'}
def stage(name,**kw):
 dump(O/'status.json',dict(stage=name,time_unix=time.time(),**kw));print(name,kw,flush=True)
def run(cmd,log):
 with Path(log).open('w') as f:subprocess.run([str(x) for x in cmd],cwd=C,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
def verify_frozen():
 for n,h in read(OLD/'implementation_bounded_freeze.json')['files'].items():assert sha(OLD_C/n)==h,n
 for n,h in read(OLD/'external_dependency_freeze.json').items():assert sha(n)==h,n
 for n,h in read(OLD/'official_rule_implementation_freeze.json')['files'].items():assert sha(n)==h,n
 for n,h in read(O/'implementation_freeze.json')['files'].items():assert sha(n)==h,n
def scene_run(s):
 dst=O/'scenes'/s;dst.mkdir(parents=True,exist_ok=False)
 native=OLD/'native'/f'{s}_400_trace'
 run(container(['/usr/bin/python3','-B','/experiment/restore_normals.py',str(native/'tsdf_final.npz'),
   str(native/'geometry/f001995.npz'),str(dst/'normals.npz')]),dst/'normals.log')
 run([P,'-B',str(C/'color_fusion.py'),s],dst/'rgb.log')
 run([P,'-B',str(C/'fill_surface.py'),s],dst/'fill.log')
 run([P,'-B',str(C/'evaluate_completion.py'),s],dst/'evaluation.log')
 return s
def audit():
 verify_frozen();reports={}
 lock=read(MAIN/'unified_eval/configs/current_protocol.json')
 assert sha(MAIN/'unified_eval/configs'/lock['config_file'])==lock['config_sha256']=='6668792cbbeff3b6a6b6f2302ad8baa1b202f463ae26306350d4617c07f601a0'
 for n,h in lock['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/n)==h
 for s in SCENES:
  dst=O/'scenes'/s;m=read(dst/'mapping_complete.json');d=read(dst/'complete.json')
  assert m['status']==d['status']=='PASS'
  assert read(dst/'normals.json')['geometry_bit_exact']
  assert read(dst/'rgb.json')['frames']==400
  assert sha(dst/'instance_surface.npz')==m['prediction_sha256']==d['prediction_sha256']
  for p,h in m['source_sha256'].items():assert sha(p)==h,p
  a=load(dst/'instance_surface.npz');b=load(OLD/'ours'/f'{s}_400_main/maps/f001995.npz')
  provenance=load(dst/'point_provenance.npz')
  np.testing.assert_array_equal(a['xyz_m'],b['xyz_m'])
  np.testing.assert_array_equal(a['instance_id'][b['instance_id']>0],b['instance_id'][b['instance_id']>0])
  change=a['instance_id']!=b['instance_id']
  np.testing.assert_array_equal(change,(b['instance_id']<=0)&(a['instance_id']>0))
  assert int(change.sum())==m['filled_points']
  np.testing.assert_array_equal(a['instance_id'][~provenance['attributes_valid']],b['instance_id'][~provenance['attributes_valid']])
  assert set(np.unique(a['instance_id'][a['instance_id']>0])).issubset(set(b['native_instance_ids']))
  original=read(OLD/'official_rule_evaluation'/s/'complete.json')
  assert d['primary']['ovimap']==original['comparison']['ovimap']
  assert d['primary']['ours_unfilled']==original['comparison']['ours']
  reports[s]={k:m[k] for k in ['filled_points','surface_points','assigned_before','assigned_after','invalid_attributes','excluded_confirmed_points']}
 out=read(O/'comparison.json')
 assert out['filled_points_total']==sum(r['filled_points'] for r in reports.values())
 result=dict(status='PASS',scenes=reports,geometry_unchanged=True,strict_votes_unchanged=True,
  existing_instance_IDs_unchanged=True,no_new_IDs=True,unknown_attributes_not_fabricated=True,
  original_holes_geodesic_settings_unchanged=True,GT_free_prediction=True,no_feedback_to_mapping=True,
  sources_and_evaluators_frozen=True,primary_and_auxiliary_same_predictions=True,
  inherited_baseline_audit=str(OLD/'final_audit.json'),result_sha256=sha(O/'comparison.json'))
 dump(O/'final_audit.json',result)
def main():
 assert read(OLD/'final_audit.json')['status']=='PASS'
 assert read(O/'completion_tests.json')['status']=='PASS'
 assert read(O/'restore_plane_test.json')['status']=='PASS'
 files={str(p):sha(p) for p in C.rglob('*') if p.is_file() and p.suffix in ['.py','.cpp','.h','.so']}
 dump(O/'implementation_freeze.json',dict(files=files,approval='User correction: hole completion is required; use existing frozen holes_geodesic',time_unix=time.time()))
 verify_frozen()
 stage('ROOM0_VALIDATION_RUNNING')
 done=[scene_run('room0')];stage('REMAINING_RUNNING',completed=done)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(scene_run,s):s for s in SCENES if s!='room0'}
  for f in concurrent.futures.as_completed(futures):
   done.append(f.result());stage('REMAINING_RUNNING',completed=done)
 stage('POOLING',completed=done)
 run([P,'-B',str(C/'evaluate_completion.py'),'aggregate'],O/'aggregate.log')
 audit();stage('COMPLETE',completed=done,comparison=read(O/'comparison.json'))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  stage('FAILED',error=repr(exc),traceback=traceback.format_exc());raise
