from pathlib import Path
import json,os,subprocess,time,traceback,sys
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019')
env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONDONTWRITEBYTECODE':'1'}
def save(name,obj):(R/name).write_text(json.dumps(obj,indent=2))
def main():
    for stage,script in [('selection','prepare_variant.py'),('E1_and_R2','e1_repair/run_selected.py'),('comparison','compare_four_way.py'),('ply','export_ply.py')]:
        save('status.json',dict(status='RUNNING',stage=stage,time=time.time()))
        with (R/(stage+'.log')).open('xb') as f:
            p=subprocess.Popen([sys.executable,'-B','-u',str(C/script)],cwd=(C/script).parent,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            assert p.wait()==0,stage
    save('complete.json',dict(status='PASS',time=time.time(),selected_masks=35))
    save('status.json',dict(status='PASS',stage='complete',time=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:save('failure.json',dict(status='FAIL',traceback=traceback.format_exc()));raise
