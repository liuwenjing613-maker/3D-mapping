from pathlib import Path
import os,sys,json
import numpy as np
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
sys.path.insert(0,str(C/'policies/main'))
from scene_completion.pipeline import write_scene_complete,finalize
scene=sys.argv[1];root=O/'main'/scene/'200/A/mapping';out=O/'published/no_repair'/scene
receipt=json.loads((root/'mapping_complete.json').read_text());assert receipt['status']=='PASS' and not receipt['repair_applied']
out.mkdir(parents=True,exist_ok=True);m=out/'scene_complete.json'
if not m.exists():
 b=root/'baseline';write_scene_complete(m,scene=scene,workflow='no_repair',frame_ids=receipt['frame_ids'],inputs=dict(surface=b/'final/instance_surface.npz',native=b/'native/instance_surface.npz',evidence=b/'surface_evidence.npz',normal_geometry=O/'attributes'/scene/'200/normals.npz'),final_processing_complete=True)
finalize(m,out/'final_B')
