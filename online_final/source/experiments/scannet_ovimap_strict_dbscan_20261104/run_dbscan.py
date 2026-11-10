from pathlib import Path
import os,sys,json,time
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
sys.path.insert(0,str(C))
import dbscan_guarded as g
scene=sys.argv[1]
os.environ.update(CVPR_SCENE=scene,CVPR_CHECKPOINT='200')
g.one('no_repair',scene)
out=O/'dbscan/no_repair'/scene;out.mkdir(parents=True,exist_ok=True)
src=O/'dbscan/base/no_repair'/scene/'merge_fill'
for n in ['instance_surface.npz','complete.json']:
 p=out/n
 if not p.exists():p.symlink_to(src/n)
