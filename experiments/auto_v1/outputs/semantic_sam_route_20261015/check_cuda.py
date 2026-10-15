import os,subprocess,json
from pathlib import Path
print('CUDA_ENV',{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_HOME']})
print('DEVICE_FILES',[(str(p),oct(p.stat().st_mode),p.stat().st_uid,p.stat().st_gid) for p in Path('/dev').glob('nvidia*')])
print('DRIVER',Path('/proc/driver/nvidia/version').read_text())
py='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
probe="import ctypes,torch; c=ctypes.CDLL('libcuda.so.1'); print('cuInit',c.cuInit(0));print('available',torch.cuda.is_available());print('device_count',torch.cuda.device_count()); print([x for x in open('/proc/self/maps').read().splitlines() if 'libcuda.so' in x][:2])"
for device in ['0','1']:
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device;env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None)
    r=subprocess.run([py,'-c',probe],env=env,capture_output=True,text=True)
    print('PROBE',device,r.returncode,r.stdout,r.stderr[-1800:])
