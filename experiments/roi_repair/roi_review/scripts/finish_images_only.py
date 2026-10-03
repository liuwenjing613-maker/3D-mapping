from pathlib import Path
import json,zipfile,sys
import numpy as np
root=Path(__file__).resolve().parents[2]
out=(root/'results/P1-A1_room0_ROI优先级_20261002').resolve()
assert out.is_relative_to(root.resolve()) and out.name=='P1-A1_room0_ROI优先级_20261002'
for p in out.rglob('*.ply'):
 assert p.resolve().is_relative_to(out)
 assert p.name.startswith('room0_ROI_') or p.name in ['context_instances.ply','context_three_states.ply']
 p.unlink()
audit=json.loads((out/'review_audit.json').read_text(encoding='utf-8'))
for key in ['ranked_three_states_PLY_sha256','numbered_boxes_PLY_sha256','all_145379_alarm_points_present_once_in_ranked_PLY']:audit.pop(key,None)
audit['all_145379_alarm_points_present_once_in_ROI_cores']=True
audit['deliverable']='ranked ROI images; PLY generation disabled at user request'
with np.load(root/'outputs/p1a1_raw_replica8_20261002/room0_final_instance_surface.npz') as f:
 for p in (out/'review_assets').glob('*/cloud.npz'):
  with np.load(p) as c:assert np.array_equal(c['xyz'],f['xyz_m'][c['point_index']])
audit['local_download_and_coordinates_verified']=True
for name in ['review_audit.json','本地核验.json']:(out/name).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
sys.path.insert(0,str(Path(__file__).parent));from assemble_review import report
ranked=json.loads((out/'roi_ranked.json').read_text(encoding='utf-8'));selection=json.loads((out/'review_selection.json').read_text(encoding='utf-8'));report(ranked,selection)
archive=out/'room0_ROI_图像与排序.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in out.rglob('*'):
  if p.is_file() and p.suffix.lower() in ['.png','.jpg','.csv','.json','.txt']:
   z.write(p,str(p.relative_to(out)))
with zipfile.ZipFile(archive) as z:assert not any(n.lower().endswith('.ply') for n in z.namelist())
old=(root/'outputs/p1a1_room0_roi_review_20261002/room0_ROI_review_assets.zip').resolve()
assert old.is_relative_to(root.resolve())
if old.exists():old.unlink()
print(json.dumps({'status':'PASS','no_new_ROI_PLY_files':not any(out.rglob('*.ply')),'image_archive_bytes':archive.stat().st_size},ensure_ascii=False))
