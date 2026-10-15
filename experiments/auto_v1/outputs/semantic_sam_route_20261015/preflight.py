from pathlib import Path
import json,subprocess

roots=[Path('/data/chenkejun/CVPR/models'),Path('/home/chenkejun/.cache/clip'),Path('/home/chenkejun/.cache/huggingface/hub')]
for r in roots:
    if not r.exists():continue
    found=[]
    for p in r.rglob('*'):
        if p.is_file() and (p.suffix in ['.pt','.pth','.bin','.safetensors'] or 'open_clip' in p.name):
            if 'CVPR/models' in str(r) or any(x in str(p).lower() for x in ['clip','vit-b','vit-h','vit-l']):
                found.append(dict(path=str(p),bytes=p.stat().st_size))
    print('WEIGHTS',str(r),json.dumps(found[:60]))
p=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009/instance_repair/frozen_e1_20261012/runtime.json')
if p.exists():print('FROZEN_RUNTIME',p.read_text()[:6500])
for p in [Path('/home/chenkejun/CVPR/AGENTS.md'),Path('/home/chenkejun/CVPR/experiments/AGENTS.md')]:
    if p.exists():print('PROJECT_INSTRUCTIONS',str(p),p.read_text())
for executable in ['/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python','/data/chenkejun/CVPR/runtime/cropformer_env/bin/python']:
    if Path(executable).exists():
        r=subprocess.run([executable,'-c',"import importlib.metadata as m,torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('open-clip',m.version('open-clip-torch'));import open_clip;print('module',open_clip.__file__)"],capture_output=True,text=True)
        print('ENV',executable,r.returncode,r.stdout,r.stderr[-1500:])
r=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.free,memory.total','--format=csv,noheader'],capture_output=True,text=True)
print('GPU',r.stdout,r.stderr)
