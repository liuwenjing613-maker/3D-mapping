"""Complete comparisons after the sequential, full mapping benchmarks finish."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')
OLD=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'

def read(p):return json.loads(p.read_text())
def write(p,x):
    q=p.with_suffix('.tmp')
    q.write_text(json.dumps(x,indent=2)+'\n')
    q.replace(p)

def command(name,args,allow_difference=False):
    write(OUT/'comparison_status.json',dict(status='RUNNING',current=name,started_unix=time.time()))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    with (OUT/(name+'.log')).open('wb') as stream:
        ret=subprocess.run([PY,'-B','-u',*map(str,args)],env=env,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT).returncode
    assert ret==0 or (allow_difference and ret==1),name
    return ret

def comparison(name,a,b):
    report=OUT/(name+'.json')
    command(name,[HERE/'compare_outputs.py',a,b,report],True)
    assert report.exists(),name+' did not produce a comparison report'
    result=read(report)
    return dict(name=name,status=result['status'],left=result['left'],right=result['right'],
        different_checks=[row for row in result['checks'] if not row['exact']])

def main():
    write(OUT/'comparison_status.json',dict(status='WAITING_FOR_BENCHMARKS'))
    while True:
        status=read(OUT/'validation_status.json')
        assert status['status']!='FAILED',status
        if status['status']=='RUNS_COMPLETE':break
        time.sleep(5)
    comparisons=[]
    comparisons.append(comparison('fresh_comparison_200_A',OUT/'reference_200_A/mapping',OUT/'checkpoint_200_A/mapping'))
    comparisons.append(comparison('original_repeat_comparison_200_A',OLD/'200/A/mapping',OUT/'reference_200_A/mapping'))
    for count,group in [(300,'A'),(200,'B'),(300,'B')]:
        comparisons.append(comparison(f'archive_comparison_{count}_{group}',OLD/str(count)/group/'mapping',OUT/f'checkpoint_{count}_{group}/mapping'))
    command('run_contracts',[HERE/'validate_run_contracts.py'])
    command('common_surface_replay',[HERE/'replay_common_surface.py'],True)
    control=read(OUT/'common_surface_comparison_200_A.json')
    timing={name:read(OUT/name/'mapping/mapping_timing.json') for name in ['reference_200_A','checkpoint_200_A']}
    old,new=timing['reference_200_A'],timing['checkpoint_200_A']
    performance={key:dict(reference_seconds=old[key],checkpoint_seconds=new[key],
        speedup=old[key]/new[key],reduction_percent=100*(1-new[key]/old[key]))
        for key in ['main_mapping_seconds','total_seconds','main_loop_including_checkpoint_seconds']}
    write(OUT/'validation_summary.json',dict(status='COMPLETE_WITH_RECORDED_COMPARISON_RESULTS',
        scope='100->200 A full fresh sequential before/after; 200/300 A/B checkpoint-only versus archives; fixed-surface kernel replay',
        common_surface=control,independent_runs=comparisons,performance=performance,
        contracts=read(OUT/'run_contracts.json'),surface_sorting_added=False,
        sampling='one fresh paired benchmark on shared server; same 4-thread configuration; no downstream repair timing'))
    write(OUT/'comparison_status.json',dict(status='COMPLETE',summary=str(OUT/'validation_summary.json')))

if __name__=='__main__':
    try:main()
    except Exception:
        write(OUT/'comparison_status.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise
