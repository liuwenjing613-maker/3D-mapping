from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
BASE=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
sys.path.insert(0,str(BASE))
from bootstrap import *
spec=importlib.util.spec_from_file_location('pure_gate_probe',BASE/'tracking100/pure_gate.py')
pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
b=load_script('_read_only50_batch',BASE/'run_batch.py')
scene,cases,objects,canonical,policy=b.inputs()
def family_view(fid,frame,families):
 z=r.load_npz(scene.records[fid]['support_file']);ol=z['mask_local_id']
 mids=[int(mid) for mid in np.unique(ol) if scene.lookup[(fid,int(mid))] in families]
 return bool(mids),(frame.mask_local>0)&~np.isin(frame.mask_local,mids)
scene.family_view=family_view
tracking=b.validate_tracking(cases)
result=pure.evaluate(b,scene,cases,objects,canonical,policy,tracking,progress=lambda n,total,seconds:print(json.dumps(dict(frames=n,total=total,seconds=seconds)),flush=True))
expected=read(R/'validated/full_track/frame_decisions.json')
assert len(result['frames'])==len(expected['frames'])==50
for got,old in zip(result['frames'],expected['frames']):
 assert got['frame']==old['frame']
 for key in ['objects','initial_objects','accepted_track_ids']:
  assert got[key]==old[key],(got['frame'],key)
 assert got['cross_case_conflict_pixels']==old.get('cross_case_conflict_pixels',0),(got['frame'],'conflict')
assert result['counts']['accepted_frames_by_track']==expected['counts']['accepted_frames_by_track']
scene.verify_unchanged()
dump(TOP/'tracking100/gate50_equivalence.json',dict(status='PASS',frames=50,objects=len(objects),all_initial_and_final_metrics_equal=True,
 accepted_masks_decisions_equal=True,pixel_conflict_counts_equal=True,seconds=result['seconds'],
 source_decisions_sha256=digest(R/'validated/full_track/frame_decisions.json'),pure_gate_sha256=digest(BASE/'tracking100/pure_gate.py'),original_gate_v1_sha256=digest(BASE/'gate_v1.py'),GT_used=False))
print('GATE50_EQUIVALENCE_PASS',flush=True)
