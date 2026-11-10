"""Queue frozen seed-only bidirectional jobs on the three verified idle GPUs."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback

ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c')
TRACK=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007/objectwise_repair_20261007/track_case.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(name,value):
    path=ROOT/name;temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def free_ram():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))/1024**2


def main():
    assert not (ROOT/'tracking_launch.json').exists()
    freeze=json.loads((ROOT/'input_freeze.json').read_text())
    assert freeze['GT_used'] is False and sha(TRACK)==freeze['tracking_code_sha256']
    assert free_ram()>=100 and shutil.disk_usage(ROOT).free/1024**3>=60
    cases=json.loads((ROOT/'seed_diagnostics.json').read_text())['cases']
    pending=list(cases);active={};completed=[];launches=[];started=time.time()
    def save_status():
        status={'status':'RUNNING' if pending or active else 'PASS','completed_cases':len(completed),
                'total_cases':len(cases),'completed':completed,'pending':[r['case_uid'] for r in pending],
                'active':[{'case_uid':job['row']['case_uid'],'pid':job['process'].pid,'gpu':gpu} for gpu,job in active.items()],
                'elapsed_seconds':round(time.time()-started,2),'source_configurations_unchanged':True}
        atomic('tracking_queue_status.json',status)
    while pending or active:
        for gpu,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            uid=job['row']['case_uid'];report_path=ROOT/'cases'/uid/'tracking/complete.json'
            if code!=0 or not report_path.is_file():
                raise RuntimeError('Tracking failed for '+uid+'; inspect '+str(job['log']))
            report=json.loads(report_path.read_text())
            assert report['status']=='PASS' and report['completed_frames']==report['total_frames']==2000
            assert report['case_config_sha256']==sha(job['row']['config_path'])
            assert report['directions_use_independent_seed_only_states'] and not report['GT_used']
            completed.append({'case_uid':uid,'gpu':gpu,'seconds':report['elapsed_seconds'],'report_sha256':sha(report_path)})
            del active[gpu]
            print(json.dumps({'event':'COMPLETE',**completed[-1]}),flush=True)
        for row in list(pending):
            cfg_path=Path(row['config_path']);cfg=json.loads(cfg_path.read_text());gpu=cfg['tracking']['cuda_visible_devices']
            if gpu in active or free_ram()<55:continue
            memory=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            if int(memory.strip())>=1024:continue
            assert sha(cfg_path)==freeze['seed_and_config_hashes'][str(cfg_path)]
            env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=gpu;env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4'
            log=ROOT/'cases'/row['case_uid']/'tracking.log'
            with log.open('ab') as stream:
                process=subprocess.Popen([cfg['tracking']['python'],str(TRACK),'--config',str(cfg_path)],
                                         stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            active[gpu]={'row':row,'process':process,'log':log}
            pending.remove(row);launches.append({'case_uid':row['case_uid'],'gpu':gpu,'pid':process.pid,
                                               'config_sha256':sha(cfg_path),'log':str(log),'started_at':time.time()})
            atomic('tracking_launch.json',{'status':'RUNNING','queue_code_sha256':sha(__file__),'tasks':launches,
                                           'excluded_gpu_3':'Previous CUDA initialization failure; leave unused',
                                           'directions':['forward','reverse'],'GT_used':False})
            print(json.dumps({'event':'LAUNCH',**launches[-1]}),flush=True)
        save_status()
        if pending or active:time.sleep(10)
    atomic('tracking_queue_complete.json',{'status':'PASS','cases':completed,'tasks':launches,
                                           'elapsed_seconds':round(time.time()-started,2),'GT_used':False})


if __name__=='__main__':
    try:main()
    except Exception:
        atomic('tracking_queue_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
