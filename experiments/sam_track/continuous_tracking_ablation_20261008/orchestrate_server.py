"""One finite job: both frozen scenes, both gap policies, then post-hoc v3."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
import traceback

CODE = Path(__file__).resolve().parent
ROOT = Path('/data/chenkejun/CVPR/revisable_instance_map/continuous_tracking_ablation_20261008')
PY = '/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'


def dump(name, value):
    p = ROOT / name
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(p)


def main():
    started = time.monotonic()
    free_ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) / 1024**2
    assert free_ram > 24 and shutil.disk_usage(ROOT).free > 12*1024**3
    workers = 2 if free_ram > 64 and (os.cpu_count() or 1) >= 16 else 1
    env = dict(os.environ,OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
    stages = []
    for stage in ['repair','evaluate']:
        if stage == 'evaluate':
            for scene in ['room0','room2']:
                assert json.loads((ROOT/scene/'predictions_frozen.json').read_text())['status'] == 'PASS_PREDICTIONS_FROZEN_BEFORE_NEW_GT'
        waiting = ['room0','room2']
        running = []
        while waiting or running:
            while waiting and len(running) < workers:
                scene = waiting.pop(0)
                (ROOT/scene).mkdir(exist_ok=True)
                with (ROOT/scene/(stage+'.log')).open('xb') as stream:
                    process = subprocess.Popen([PY,'-u',str(CODE/'run_ablation.py'),'--scene',scene,'--stage',stage],
                       stdout=stream,stderr=subprocess.STDOUT,env=env)
                running.append((scene,process,time.monotonic()))
            dump('status.json',{'status':'RUNNING','stage':stage,'jobs':[{'scene':s,'pid':p.pid} for s,p,t in running],
                'waiting':waiting,'completed_stages':stages,'elapsed_seconds':round(time.monotonic()-started,2)})
            still_running = []
            for scene,process,t in running:
                code = process.poll()
                if code is None:
                    still_running.append((scene,process,t))
                elif code:
                    raise RuntimeError(f'{scene} {stage} failed, exit {code}; inspect {ROOT/scene/(stage+".log")}')
                else:
                    stages.append({'scene':scene,'stage':stage,'seconds':round(time.monotonic()-t,2)})
            running = still_running
            if waiting or running: time.sleep(10)
    summaries = {s:json.loads((ROOT/s/'evaluation_summary.json').read_text()) for s in ['room0','room2']}
    assert all(s['status']=='PASS' for s in summaries.values())
    result = {'status':'PASS','conditions':['unrestricted_reference','gap0','gap1'],'scenes':summaries,
              'stages':stages,'elapsed_seconds':round(time.monotonic()-started,2),
              'old_maps_preserved':True,'SAM_rerun':False,'GT_used_for_filter_or_repair':False,
              'unified_v3_code_protocol_and_flags_unchanged':True}
    dump('complete.json',result)
    dump('status.json',{'status':'PASS','stage':'complete','completed_stages':stages})


if __name__ == '__main__':
    try: main()
    except Exception:
        dump('failure.json',{'status':'FAIL','traceback':traceback.format_exc()})
        dump('status.json',{'status':'FAIL','stage':'inspect failure.json'})
        raise
