from prepare_inputs import C,O,dump,sha
import json
import sys
for scene in (sys.argv[1:] or ['scene0011_00','scene0050_00']):
 d=O/'inputs'/scene;r=json.loads((d/'selection.json').read_text());m=d/'masks';m.mkdir(exist_ok=True);rows=[]
 for row in r['rows']:
  fid,raw=row['frame'],row['original_frame'];src=d/'raw_masks'/f'{raw}.png';assert src.exists()
  out=m/f'frame{fid:06d}.png'
  if not out.exists():out.symlink_to(src)
  rec=json.loads((d/'raw_masks/records'/f'{raw}.json').read_text());rec.update(frame=fid,original_frame=raw,mask_sha256=sha(src));rows.append(rec)
 (m/'frames.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
 dump(d/'complete.json',dict(status='PASS',frames=len(rows),selection_sha256=sha(d/'selection.json')))
