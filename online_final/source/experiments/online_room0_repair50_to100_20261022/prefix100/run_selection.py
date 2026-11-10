"""Run the accepted upstream and swapped filtering order on prefix-only inputs."""
from online_common import *
import argparse
import os
import subprocess
import time
import traceback

ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1', 'OPENBLAS_NUM_THREADS':'4', 'OMP_NUM_THREADS':'4',
       'CUDA_DEVICE_ORDER':'PCI_BUS_ID', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
       'PYTHONPATH':str(C/'guard')+':'+str(C)+':/home/chenkejun/CVPR/repos/sam2_1',
       'CVPR_ONLINE_GUARD':str(R/'input_guard.json'),
       'XDG_CACHE_HOME':str(R/'runtime_cache'), 'CUDA_CACHE_PATH':str(R/'cuda_cache'),
       'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09'}

def launch(name, script, args=(), python=PY, device=None):
    env = dict(ENV)
    if device: env['CUDA_VISIBLE_DEVICES'] = device
    path = C / script
    logfile = R / (name + '.log')
    with logfile.open('ab') as log:
        log.write(('\nSTART '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n').encode())
        proc = subprocess.Popen([python,'-B','-u',str(path),*map(str,args)],cwd=path.parent,
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write(R/'selection_progress.json',dict(stage=name,pid=proc.pid,gpu=device,started=time.time(),log=str(logfile)))
    print('STAGE',name,proc.pid,flush=True)
    return proc

def run(name, script, args=(), python=PY, device=None):
    proc = launch(name,script,args,python,device)
    status = proc.wait()
    if status: raise RuntimeError(name+' failed, exit '+str(status))

def devices():
    rows=[]
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    for line in output.splitlines():
        uid,free,util=[s.strip() for s in line.split(',')]
        if int(free)>14000: rows.append((int(free),-int(util),uid))
    rows.sort(reverse=True)
    working=[]
    for _,_,uid in rows:
        check=subprocess.run([PY,'-B','-c','import torch; x=torch.zeros(1,device="cuda")'],
            env={**ENV,'CUDA_VISIBLE_DEVICES':uid},capture_output=True,timeout=40)
        if check.returncode==0:working.append(uid)
        if len(working)==2:break
    assert working,'No verified working GPU with sufficient free memory'
    return working

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',default='catalog')
    args=parser.parse_args()
    assert read(R/'mapping_complete.json')['frame_ids']==list(FIDS)
    order=['catalog','prepare','models','summary','overmerge','intersection','versions','semantic_classify',
           'semantic_cabinet','semantic_route','grouping_r2','grouping_r3','v1','semantic_background','v2']
    assert args.start in order
    steps=order[order.index(args.start):]
    for step in steps:
        if step=='catalog':run(step,'catalog/build_candidates.py',['room0'])
        elif step=='prepare':run(step,'frontend/prepare_inputs.py')
        elif step=='models':
            gpu=devices()
            if len(gpu)==1:
                run('cropformer','frontend/infer_cropformer.py',python=CROP,device=gpu[0])
                run('sam','frontend/infer_sam.py',python=SAM,device=gpu[0])
            else:
                a=launch('cropformer','frontend/infer_cropformer.py',python=CROP,device=gpu[0])
                b=launch('sam','frontend/infer_sam.py',python=SAM,device=gpu[1])
                write(R/'selection_progress.json',dict(stage='models',pids=[a.pid,b.pid],gpus=gpu,started=time.time()))
                sa,sb=a.wait(),b.wait()
                assert sa==sb==0, ('inference failed',sa,sb)
        elif step=='summary':run(step,'frontend/summarize_frontend.py')
        elif step in ['overmerge','intersection','versions','grouping_r2','grouping_r3','v1']:
            run(step,step+('/score.py' if step=='overmerge' else '/compute.py'))
        elif step=='semantic_classify':run(step,'semantic_route/classify.py',device=devices()[0])
        elif step=='semantic_cabinet':run(step,'semantic_route/refine_cabinet.py',device=devices()[0])
        elif step=='semantic_route':run(step,'semantic_route/route.py')
        elif step=='semantic_background':
            run(step,'v2/semantic/run.py',['--manifest',R/'v1/seed_manifest.json','--raw-config',C/'config.json',
                '--out',R/'v2/semantic','--config',C/'v2/semantic/config.json'],device=devices()[0])
        elif step=='v2':run(step,'v2/measure_and_select.py')
    complete=read(R/'v2/complete.json')
    write(R/'selection_complete.json',dict(status='PASS',frame_ids=list(FIDS),last_frame_id=495,
        seed_pipeline='auto-v3-swap on prefix-only evidence',result=complete,repair_applied=True, first_repair_processed_frame=50, second_repair_applied=False,GT_used=False))
    write(R/'selection_progress.json',dict(stage='complete',result=complete,completed=time.time()))

if __name__=='__main__':
    try:main()
    except Exception:
        write(R/'selection_failure.json',dict(error=traceback.format_exc(),time=time.time()))
        raise
