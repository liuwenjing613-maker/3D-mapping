"""Aggregate all 32 final-B maps with one R2 geometry/GT/correspondence per scene."""
from finish_scene import *
VARIANTS=['no_repair','main','online_A','online_B']
def main():
 lock=protocol()
 for variant in VARIANTS:
  rows=[]
  for scene in SCENES:
   ev=O/'evaluation'/variant/scene;receipt=read(ev/'complete.json')
   assert receipt['status']=='PASS' and sha(receipt['prediction'])==receipt['prediction_sha256']
   m=read(ev/'r2/adapter_manifest.json')
   rows.append(dict(gt=m['gt_file'],prediction=str(ev/'r2/canonical_prediction.npz'),diagnostic_prediction=str(ev/'r2/diagnostic_support_prediction.npz')))
  out=O/'pooled'/variant;out.mkdir(parents=True,exist_ok=True);dump(out/'batch.json',dict(scenes=rows))
  run(['eval-batch','--scenes',out/'batch.json','--out',out],out/'run.log')
  summary=read(out/'summary.json');assert summary['CA_PRF1_0_5']['TP']+summary['CA_PRF1_0_5']['FN']==350
 for scene in SCENES:
  ref=read(O/'evaluation/no_repair'/scene/'complete.json')
  a=load(ref['prediction'])
  for variant in VARIANTS[1:]:
   r=read(O/'evaluation'/variant/scene/'complete.json');b=load(r['prediction'])
   for field in ['xyz_m','rgb']:np.testing.assert_array_equal(a[field],b[field])
   assert r['shared_correspondence_sha256']==ref['shared_correspondence_sha256']
   dest=O/'pairs'/scene/('no_repair_to_'+variant);dest.mkdir(parents=True,exist_ok=True)
   gt=lock['canonical_GT'][scene]['file']
   run(['eval-repair-pair','--gt',gt,'--before',O/'evaluation/no_repair'/scene/'r2/canonical_prediction.npz','--after',O/'evaluation'/variant/scene/'r2/canonical_prediction.npz','--out',dest],dest/'run.log')
 result=dict(status='PASS',scenes=SCENES,variants=VARIANTS,upstream=read(O/'plan.json')['upstream'],source_lock_sha256=sha(C/'source_lock.json'),same_geometry=True,same_rgb=True,same_GT_and_correspondence=True,final_B_once=True,ABC_non_window_A_only=True,R2_GT=350,frozen=False,pooled={v:read(O/'pooled'/v/'summary.json') for v in VARIANTS})
 dump(O/'comparison.json',result);dump(O/'final_audit.json',{k:v for k,v in result.items() if k!='pooled'})
 print('ALL_32_RESULTS_PASS',flush=True)
if __name__=='__main__':main()
