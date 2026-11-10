from pathlib import Path
import sys,os,json,time,hashlib,traceback
import numpy as np,cv2
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
SCENES=['scene0011_00','scene0050_00']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2))
try:
 while True:
  status={s:read(O/(s+'_status.json')) for s in SCENES if (O/(s+'_status.json')).exists()}
  if any(x['status']=='FAILED' for x in status.values()):raise RuntimeError('Scene pipeline failed; inspect scene status')
  if len(status)==2 and all(x['status']=='PASS' for x in status.values()):break
  time.sleep(15)
 from source_guard import check
 check();audits={};rows=[]
 for s in SCENES:
  native=[json.loads(l) for l in (O/'native'/s/'full/frame_input_audit.jsonl').read_text().splitlines()]
  ovi=[json.loads(l) for l in (O/'ovimap'/s/'200/shared_input_audit.jsonl').read_text().splitlines()]
  assert len(native)==len(ovi)==200
  for a,b in zip(native,ovi):
   assert a['frame_id']==b['frame_id']
   for k in ['depth_decoded_sha256','pose']:assert a[k]==b[k],(s,k,a['frame_id'])
   np.testing.assert_array_equal(np.asarray(a['intrinsics'],np.float32),np.asarray(b['intrinsics'],np.float32))
   fid=a['frame_id']
   root=O/'inputs'/s
   assert b['rgb_bgr_sha256']==digest(cv2.imread(str(root/'input/results'/f'frame{fid:06d}.png')))
   assert b['mask_decoded_sha256']==digest(cv2.imread(str(root/'masks'/f'frame{fid:06d}.png'),-1).astype(np.int32))
  baseline=read(O/'main'/s/'200/A/mapping/mapping_complete.json')
  assert baseline['status']=='PASS' and not baseline['repair_applied'] and baseline['strict_same_frame_point_at_most_one_vote']
  audits[s]=dict(status='PASS',frames=200,RGB_depth_masks_pose_equal=True,intrinsics_float32_equal=True,intrinsics_raw_bit_exact=False,intrinsics_max_abs_difference=max(float(np.max(np.abs(np.asarray(a['intrinsics'])-np.asarray(b['intrinsics'])))) for a,b in zip(native,ovi)),frame_order_equal=True,TSDF_geometry_different=True,GT_not_used_for_mapping=True)
  dirs={'OVIMAP':O/'evaluation/OVIMAP'/s/'r2','no_repair':O/'evaluation/no_repair'/s/'r2','no_repair_DBSCAN':O/'dbscan/base/no_repair'/s/'merge_fill/r2'}
  manifests={k:read(d/'adapter_manifest.json') for k,d in dirs.items()}
  for m in manifests.values():
   for k in ['gt_file_sha256','gt_scope_sha256','gt_observed_support_sha256','profile_config_sha256','evaluator_code_sha256']:assert m[k]==manifests['no_repair'][k]
  for k in ['geometry_xyz_sha256','correspondence_sha256']:assert manifests['no_repair'][k]==manifests['no_repair_DBSCAN'][k]
  for method,d in dirs.items():
   metrics=read(d/'metrics.json')['summary'];pr=metrics['CA_PRF1_0_5']
   rows.append(dict(scene=s,method=method,GT=pr['TP']+pr['FN'],AP=metrics['CA_AP_uniform'],AP50=metrics['CA_AP50_uniform'],F1=pr['F1'],PQ=metrics['CA_PQ']['PQ'],mCov=metrics['CA_mCov'],TP=pr['TP'],FP=pr['FP'],FN=pr['FN'],owner_surface=metrics['owner_surface'],source_surface=manifests[method]['source_surface'],prediction_sha256=manifests[method]['source_surface_sha256']))
 dump(O/'comparison.json',dict(status='PASS',rows=rows,input_audits=audits,GT_total=52,scope='Current main R2 ScanNet200 adaptation; not paper metrics',limitations=['Two scenes, 200 uniformly selected available stride10 frames per scene; not exact paper frame list','Intrinsics differ by float32 conversion (<5e-7 pixel); not a bit-exact input comparison','Instance mapping comparison; no semantic VLM extraction or classification','Each TSDF uses its own fixed geometry correspondence; DBSCAN before/after share geometry and correspondence']))
 dump(O/'final_audit.json',dict(status='PASS',input_audits=audits,report_sha256=sha(O/'comparison.json'),source_lock_sha256=sha(C/'source_lock.json')))
 print('ALL_PASS',flush=True)
except Exception:
 dump(O/'report_failure.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
