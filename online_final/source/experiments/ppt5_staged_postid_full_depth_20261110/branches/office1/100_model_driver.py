from common_continue import *
from selection_runtime import run
import sys,traceback
stage=sys.argv[1];sc=CONT/'selection_code/100/B';out=CANDIDATE_ROOT/'frontend'/('run_'+stage)
script=sc/'frontend'/('infer_'+('sam' if stage=='sam3' else 'cropformer')+'.py')
python='/data/chenkejun/CVPR/envs/dualmap_torch27/bin/python' if stage=='sam3' else CROP
try:
 row=run(stage,script,python=python,gpu=True,out=out);dump(out/'driver_status.json',dict(status='PASS',process=row,tracking_or_repair=False))
except Exception:
 dump(out/'driver_status.json',dict(status='FAILED',traceback=traceback.format_exc(),tracking_or_repair=False));raise
