set -e
python3 - <<'PY'
from pathlib import Path
import re,subprocess
out=Path('/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09');out.mkdir(parents=True,exist_ok=True)
for p in Path('/usr/lib/x86_64-linux-gnu').glob('*.so.580.126.09'):
 text=subprocess.run(['readelf','-d',str(p)],capture_output=True,text=True).stdout
 m=re.search(r'\(SONAME\).*?\[(.*?)\]',text)
 for name in {p.name,m.group(1) if m else p.name}:
  link=out/name
  if not link.exists():link.symlink_to(p)
 print(p.name,'SONAME',m.group(1) if m else '')
print('DRIVER_LINK_DIR',out)
PY
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09 /data/chenkejun/ovimap_runtime_20260908/envs/cropformer/bin/python - <<'PY'
import torch
print('CUDA_OK',torch.cuda.is_available(),torch.cuda.get_device_name(0))
PY
