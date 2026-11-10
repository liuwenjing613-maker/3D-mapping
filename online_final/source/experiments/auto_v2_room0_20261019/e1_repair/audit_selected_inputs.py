from bootstrap import C,R,S,digest,verify_sources
import json
import numpy as np
from PIL import Image
def main():
 verify_sources();selection=json.loads((C/'confirmed_selection.json').read_text());byid={x['original_track_id']:x for x in selection['included']}
 actual=json.loads((R/'seed_id_map.json').read_text())['objects'];assert {x['track_id'] for x in actual}==set(byid)
 cases=json.loads((R/'seed_manifest.json').read_text())['seeds'];seen=set()
 for c in cases:
  lab=np.asarray(Image.open(R/c['seed_file']))
  for o in c['objects']:
   tid=o['track_id'];s=byid[tid];assert tid not in seen;seen.add(tid)
   p=S/s['mask_relative_path'];assert digest(p)==s['mask_sha256']
   assert np.array_equal(lab==tid,np.asarray(Image.open(p))>0)
 assert seen==set(byid)
 (R/'selected_input_audit.json').write_text(json.dumps(dict(status='PASS',original_seed_pixels_verified=len(seen),original_track_ids_preserved=True,manual_overrides=[],original_E1_sources_verified=True)))
 print('INPUT_AUDIT_PASS',len(seen),flush=True)
if __name__=='__main__':main()
