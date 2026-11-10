from prepare_attributes import *
import sys,cv2
scene=sys.argv[1]
rows=[json.loads(x) for x in (O/'native'/scene/'full/frame_input_audit.jsonl').read_text().splitlines()]
for row in rows:
 fid=row['frame_id'];mask=cv2.imread(str(O/'inputs'/scene/'masks'/f'frame{fid:06d}.png'),-1);assert mask.shape==(480,640)
 row['mask_decoded_sha256']=digest(mask.astype(np.int32));dump(O/'frame_meta'/scene/f'f{fid:06d}.json',row)
import time
while not (O/'attributes'/scene/'200/complete.json').exists():
 pid={'scene0011_00':1790430,'scene0050_00':1790432}[scene]
 if not (Path('/proc')/str(pid)).exists():raise RuntimeError('Early attribute producer exited before completion')
 time.sleep(5)
attributes(scene,200)
