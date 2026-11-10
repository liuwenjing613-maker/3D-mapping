from pathlib import Path
import hashlib,json,os,subprocess,sys
import numpy as np
C=Path(__file__).resolve().parent
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
SCENES=['room0','room1','room2','office0','office1','office2','office3','office4']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def run(cmd,log):
 with Path(log).open('w') as f:
  subprocess.run([str(x) for x in cmd],cwd=MAIN,env={**os.environ,'PYTHONPATH':str(MAIN),'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'},stdout=f,stderr=subprocess.STDOUT,check=True)
def validate_scene(scene):
 native=O/'native'/f'{scene}_400_trace';ours=O/'ours'/f'{scene}_400_main'
 assert read(native/'readout_complete.json')['processed_frames']==400
 assert read(ours/'complete.json')['processed_frames']==400
 a=[json.loads(x) for x in (native/'prefix_audit.jsonl').read_text().splitlines()]
 b=[json.loads(x) for x in (ours/'audit.jsonl').read_text().splitlines()]
 assert len(a)==len(b)==400
 for i,(x,y) in enumerate(zip(a,b)):
  assert x['frame_id']==y['frame_id']==i*5
  assert x['geometry_sha256']==y['geometry_sha256']
  assert x['readonly_verified'] and y['strict_vote_verified']
 for count in [1,2,5,10,20,50,100,200,300,400]:assert b[count-1]['full_replay_verified']
 surfaces={'ours':ours/'maps/f001995.npz','ovimap':native/'ovi/f001995.npz'}
 with np.load(surfaces['ours']) as x,np.load(surfaces['ovimap']) as y:
  np.testing.assert_array_equal(x['xyz_m'],y['xyz_m'])
 report={'status':'PASS','scene':scene,'frames':400,'same_geometry_all_prefixes':True,
  'readonly_all_prefixes':True,'strict_votes_all_prefixes':True,'no_repair':True,'no_hole_fill':True,
  'input_audit_sha256':sha(ours/'audit.jsonl'),'native_audit_sha256':sha(native/'prefix_audit.jsonl'),
  'prediction_sha256':{k:sha(v) for k,v in surfaces.items()}}
 dump(O/'verification'/f'{scene}.json',report)
 return surfaces,report
def fields(s):
 return {'PQ':s['CA_PQ']['PQ'],'F1':s['CA_PRF1_0_5']['F1'],'AP50':s['CA_AP50_uniform'],'AP':s['CA_AP_uniform'],'mCov':s['CA_mCov'],**{k:s['CA_PRF1_0_5'][k] for k in ['TP','FP','FN']}}
def evaluate(scene):
 surfaces,report=validate_scene(scene)
 lock=read(MAIN/'unified_eval/configs/current_protocol.json')
 # Verify lock and current evaluator through its official entry point.
 run([sys.executable,'-B','-m','unified_eval.cli','current-protocol'],O/'verification'/f'{scene}_protocol.log')
 config=MAIN/'unified_eval/configs'/lock['config_file'];assert sha(config)==lock['config_sha256']
 for f,h in lock['evaluator_code_sha256'].items():assert sha(MAIN/'unified_eval'/f)==h
 gt=Path(lock['canonical_GT'][scene]['file']);assert sha(gt)==lock['canonical_GT'][scene]['file_sha256']
 from hashlib import sha256
 # Compute actual frame list digest in the same format as the fixed GT contract.
 audit=read(Path('/data/chenkejun/CVPR/results/v3_object_observed_repair_audit_20261009')/scene/'OVI-MAP_full400/adapter_manifest.json')
 frame_digest=audit['source_provenance']['frame_list_sha256']
 records=[json.loads(x) for x in (O/'native'/f'{scene}_400_trace'/'prefix_audit.jsonl').read_text().splitlines()]
 assert [r['frame_id'] for r in records]==list(range(0,2000,5))
 dst=O/'evaluation'/scene;dst.mkdir(parents=True,exist_ok=True)
 result={}
 for method,surface in surfaces.items():
  provenance={'input_scope_verified':True,'frame_count':400,'frame_list_sha256':frame_digest,
   'alignment_basis':'all 400 raw depth/mask decoding and actual OVI pose/intrinsics verified; identical geometry arrays at every prefix',
   'alignment_audit_file':str(O/'verification'/f'{scene}.json'),'alignment_audit_sha256':sha(O/'verification'/f'{scene}.json'),
   'prediction_frozen_before_GT':True,'source_map_sha256':sha(surface),'experiment':'ovimap_tsdf_strict_20261031',
   'geometry':'native OVI incremental TSDF','publication':'native strict >=2 and >=0.67' if method=='ours' else 'native integer IDs with same-coordinate consensus',
   'implementation_manifest':str(O/'implementation_bounded_freeze.json'),'implementation_manifest_sha256':sha(O/'implementation_bounded_freeze.json'),'GT_used_for_mapping':False,'repair_enabled':False,'hole_completion_enabled':False}
  pf=dst/(method+'_provenance.json');dump(pf,provenance)
  output=dst/method
  run([sys.executable,'-B','-m','unified_eval.cli','adapt-surface','--gt',gt,'--surface',surface,
    '--fixed-correspondence',dst/'shared_correspondence.npz','--source-provenance',pf,
    '--method-name',method+'_ovimap_incremental_tsdf','--method-commit',sha(C/'baseline_bounded.py'),
    '--out',output],dst/(method+'.log'))
  m=read(output/'adapter_manifest.json')
  assert m['evaluator_code_sha256']==lock['evaluator_code_sha256']
  assert m['profile_config_sha256']==lock['config_sha256']
  assert m['gt_file_sha256']==lock['canonical_GT'][scene]['file_sha256']
  assert sha(surface)==report['prediction_sha256'][method]
  result[method]=fields(read(output/'metrics.json')['summary'])
 a=read(dst/'ours/adapter_manifest.json');b=read(dst/'ovimap/adapter_manifest.json')
 for k in ['correspondence_sha256','correspondence_file_sha256','gt_file_sha256']:assert a[k]==b[k]
 dump(dst/'comparison.json',result);print(json.dumps({'scene':scene,**result}),flush=True)
def pool():
 dst=O/'pooled';dst.mkdir(exist_ok=True);results={}
 for method in ['ours','ovimap']:
  rows=[]
  for scene in SCENES:
   d=O/'evaluation'/scene/method;m=read(d/'adapter_manifest.json')
   rows.append({'gt':m['gt_file'],'prediction':str(d/'canonical_prediction.npz'),'diagnostic_prediction':str(d/'diagnostic_support_prediction.npz')})
  target=dst/method;target.mkdir(exist_ok=True);dump(target/'batch.json',{'scenes':rows})
  run([sys.executable,'-B','-m','unified_eval.cli','eval-batch','--scenes',target/'batch.json','--out',target],target/'eval.log')
  results[method]=fields(read(target/'summary.json'))
 dump(O/'comparison.json',results)
if __name__=='__main__':
 if sys.argv[1]=='pool':pool()
 else:evaluate(sys.argv[1])
