"""Configuration shared by the explicitly approved prefix-50 experiment."""
from pathlib import Path
import hashlib
import json

NAME = 'online_room0_prefix50_20261021'
C = Path('/home/chenkejun/CVPR/experiments') / NAME
R = Path('/data/chenkejun/CVPR/results') / NAME
SNAPSHOT = Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot')
FIDS = tuple(range(0, 246, 5))
PY = '/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
SAM = '/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python'
CROP = '/data/chenkejun/CVPR/runtime/cropformer_env/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
