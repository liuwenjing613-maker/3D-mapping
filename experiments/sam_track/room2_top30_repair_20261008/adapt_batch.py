"""Reuse the verified room0 pipeline with explicit, audited room2 batch changes."""
from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[1]
SOURCE=WORK/'outputs/room0_human_repair_20261007'
annotation=WORK/'room2_前30_修复候选选择_20261008.json'
digest=hashlib.sha256(annotation.read_bytes()).hexdigest();batch='batch_'+digest[:12]
ROOT='/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/'+batch
CODE='/home/chenkejun/CVPR/experiments/room2_top30_repair_20261008/'+batch
oldroot='/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d'
def save(name,text):
 with (HERE/name).open('w',encoding='utf-8',newline='\n') as s:s.write(text)
def once(text,a,b):
 assert text.count(a)==1,(a,text.count(a))
 return text.replace(a,b,1)
changes=[]
for name in ['prepare_server.py','cross_seed_diagnostics.py','run_batch.py','evaluate_batch.py','tracking_queue.py']:
 text=(SOURCE/name).read_text(encoding='utf-8').replace(oldroot,ROOT).replace('room0','room2')
 if name=='prepare_server.py':
  text=once(text,"scene=r.Scene('room2'); scene.baseline_control()","scene=r.Scene('room2'); scene.baseline_control()\n    assert scene.source_hashes[str(r.BASE/'final/room2/P1-A1/diffusion/holes_geodesic/instance_surface.npz')]==ingest['source_surface_sha256']\n    page=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_selection_top30_20261008/review')\n    for row in ingest['seeds']:\n        if row['status']=='READY':\n            assert r.sha(page/row['source_label_relative_path'])==row['source_label_sha256']")
  text=once(text,"'repair_cases':len(diagnostics),'selected_masks':len(supports)","'repair_cases':len(diagnostics),'selected_masks':len(supports),'total_ranked_cases':30,'unannotated_cases':ingest['unannotated_cases']")
 if name=='run_batch.py':
  text=once(text,"'no_reliable_seed_cases':14-len(cases)","'no_reliable_seed_cases':json.loads((ROOT/'inputs/local_ingest.json').read_text())['no_reliable_seed_cases'],'total_ranked_cases':30,'unannotated_cases':json.loads((ROOT/'inputs/local_ingest.json').read_text())['unannotated_cases']")
 if name=='tracking_queue.py':
  text=once(text,"'excluded_gpu_3':'Previous room2 CUDA initialization failure; leave unused'","'excluded_gpu_3':'Previous CUDA initialization failure; leave unused'")
 save(name,text)
 changes.append({'file':name,'source_sha256':hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(),'adapted_sha256':hashlib.sha256((HERE/name).read_bytes()).hexdigest()})
for name in ['joint_batch_ops.py','test_joint_batch_ops.py']:
 shutil.copyfile(SOURCE/name,HERE/name)
export=(SOURCE/'export_review_v2.py').read_text(encoding='utf-8').replace(oldroot,ROOT).replace('room0','room2')
export=export.replace('room2_repair_batch_20261007','room2_top30_repair_20261008')
export=once(export,"'annotation_sha256':complete['annotation_sha256']","'annotation_sha256':complete['annotation_sha256'],'unannotated_cases':json.loads((ROOT/'inputs/local_ingest.json').read_text())['unannotated_cases'],'total_ranked_cases':30")
save('export_review.py',export)
save('task_paths.json',json.dumps({'scene':'room2','annotation_sha256':digest,'batch':batch,'ROOT':ROOT,'CODE':CODE},indent=2)+'\n')
save('adaptation_receipt.json',json.dumps({'status':'PASS','sources':changes,'scope':'partial human export: 18 decisions, 14 groups, 25 masks, 12 unannotated','numerical_thresholds_changed':False,'vote_algorithm_changed':False,'baseline_changed':False,'changes':['new room2 input/source hashes and isolated batch paths','partial annotation counts and explicit unannotated cases','use already corrected color palette exporter']},indent=2)+'\n')
print(json.dumps({'ROOT':ROOT,'CODE':CODE,'annotation_sha256':digest}))
