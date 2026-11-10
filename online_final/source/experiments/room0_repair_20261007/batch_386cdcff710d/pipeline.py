"""Run frozen room0 repair, then unchanged v3 evaluation and real PLY export."""
from pathlib import Path
import hashlib,json,os,subprocess,time,traceback

CODE=Path(__file__).resolve().parent
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d')
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(name,value):
    p=ROOT/name;temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)

def main():
    started=time.monotonic()
    track=json.loads((ROOT/'tracking_queue_complete.json').read_text())
    assert track['status']=='PASS'
    alias=json.loads((ROOT/'alias_review.json').read_text())
    assert alias['status']=='FROZEN' and alias['GT_used'] is False
    assert json.loads((ROOT/'global_association.json').read_text())['status']=='PASS'
    codes={str(CODE/name):sha(CODE/name) for name in
           ['pipeline.py','run_batch.py','joint_batch_ops.py','evaluate_batch.py','export_review.py']}
    write('pipeline_freeze.json',{'status':'FROZEN','code_sha256':codes,
          'alias_review_sha256':sha(ROOT/'alias_review.json'),
          'tracking_queue_complete_sha256':sha(ROOT/'tracking_queue_complete.json')})
    stages=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='8',OMP_NUM_THREADS='8')
    for stage,script,receipt in [('repair','run_batch.py','repair_complete.json'),
           ('evaluation','evaluate_batch.py','evaluation_summary.json'),
           ('export','export_review.py','review_bundle.json')]:
        for path,digest in codes.items():assert sha(path)==digest
        assert sha(ROOT/'alias_review.json')==json.loads((ROOT/'pipeline_freeze.json').read_text())['alias_review_sha256']
        t=time.monotonic();log=ROOT/(stage+'.log')
        with log.open('xb') as stream:
            process=subprocess.Popen([PY,'-u',str(CODE/script)],stdout=stream,stderr=subprocess.STDOUT,env=env)
            write('pipeline_status.json',{'status':'RUNNING','stage':stage,'pid':process.pid,
                  'elapsed_seconds':round(time.monotonic()-started,2),'log':str(log),'completed_stages':stages})
            code=process.wait()
        assert code==0, f'{stage} exit {code}; inspect {log}'
        result=json.loads((ROOT/receipt).read_text());assert result['status']=='PASS'
        stages.append({'stage':stage,'seconds':round(time.monotonic()-t,2),'receipt_sha256':sha(ROOT/receipt)})
    write('pipeline_complete.json',{'status':'PASS','stages':stages,'elapsed_seconds':round(time.monotonic()-started,2)})
    write('pipeline_status.json',{'status':'PASS','stage':'complete','completed_stages':stages})

if __name__=='__main__':
    try:main()
    except Exception:
        write('pipeline_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
