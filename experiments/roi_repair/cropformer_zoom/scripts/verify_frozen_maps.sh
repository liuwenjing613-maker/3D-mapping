set -e
python3 - <<'PY'
from pathlib import Path
import hashlib,json
R=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002')
expected={'native/room0/P1-A1/surface_p0/surface_evidence.npz':'6066ad83c20b0120541f13d96033eecad172357459f5453389a8bb63d84c83ce','final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz':'156755555eec78c2f7177f4a2be7602466e8ab9d1d4c020fb99628054a997666'}
for file,digest in expected.items():
 h=hashlib.sha256()
 with (R/file).open('rb') as f:
  for a in iter(lambda:f.read(2**20),b''):h.update(a)
 assert h.hexdigest()==digest
out=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003')
(out/'frozen_map_hash_audit.json').write_text(json.dumps({'unchanged':True,'verified_source_hashes':expected},indent=2)+'\n')
print('Frozen native evidence and final map hashes unchanged')
PY
