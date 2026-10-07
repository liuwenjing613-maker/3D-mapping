"""Recover a palette-only display failure, preserving all frozen predictions."""
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
    p=ROOT/name;tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
    started=time.monotonic()
    assert not (ROOT/'export_recovery_complete.json').exists()
    old=json.loads((ROOT/'pipeline_freeze.json').read_text())
    for path,digest in old['code_sha256'].items():assert sha(path)==digest
    repair=json.loads((ROOT/'repair_complete.json').read_text());assert repair['status']=='PASS'
    evaluation=json.loads((ROOT/'evaluation_summary.json').read_text());assert evaluation['status']=='PASS'
    outputs={}
    for name in ['seed_only','full_track']:
        done=json.loads((ROOT/'validated'/name/'complete.json').read_text());assert done['status']=='PASS'
        outputs.update(done['output_sha256'])
    outputs.update({str(ROOT/name):sha(ROOT/name) for name in ['repair_complete.json','evaluation_summary.json','evaluation_freeze.json']})
    for path,digest in outputs.items():assert sha(path)==digest
    frozen={'status':'FROZEN','only_change':'Allocate rendering palette for every tracked associated ID, including IDs absent from final label arrays',
       'original_export_sha256':sha(CODE/'export_review.py'),'fixed_export_sha256':sha(CODE/'export_review_v2.py'),
       'recovery_code_sha256':sha(__file__),'prediction_and_evaluation_sha256':outputs,
       'prior_failure_sha256':sha(ROOT/'pipeline_failure.json'),'GT_repair_parameters_changed':False}
    write('export_recovery_freeze.json',frozen)
    with (ROOT/'export_v2.log').open('xb') as log:
        process=subprocess.Popen([PY,'-u',str(CODE/'export_review_v2.py')],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'))
        write('pipeline_status.json',{'status':'RUNNING','stage':'export_recovery','pid':process.pid,'prior_failure_preserved':True})
        code=process.wait()
    assert code==0, 'Inspect export_v2.log'
    for path,digest in outputs.items():assert sha(path)==digest
    for path,digest in old['code_sha256'].items():assert sha(path)==digest
    bundle=json.loads((ROOT/'review_bundle.json').read_text());assert bundle['status']=='PASS' and bundle['PLY_roundtrip_verified']
    receipt={'status':'PASS','seconds':round(time.monotonic()-started,2),'frozen_maps_and_metrics_unchanged':True,
       'prior_failure_preserved':True,'freeze_sha256':sha(ROOT/'export_recovery_freeze.json'),'review_bundle_sha256':bundle['sha256']}
    write('export_recovery_complete.json',receipt)
    status=json.loads((ROOT/'pipeline_status.json').read_text())
    completed={'status':'PASS','repair_and_evaluation_complete':True,'recovered_stage':'export',
               'export_recovery':receipt,'prior_pipeline_failure_preserved':True}
    write('pipeline_complete.json',completed);write('pipeline_status.json',{'status':'PASS','stage':'complete',**completed})

if __name__=='__main__':
    try:main()
    except Exception:write('export_recovery_failure.json',{'status':'FAIL','traceback':traceback.format_exc()});raise
