"""Explicit ScanNet dataset adapter; unchanged current-main R2 config and scoring sources."""
from pathlib import Path
import os,sys,json,argparse
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009');sys.path.insert(0,str(MAIN))
from unified_eval.current_protocol import CURRENT_CONFIG,require_current_protocol
from unified_eval.repair_profile import adapt_surface
from unified_eval.io import sha256_file,repair_evaluator_code_hashes
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['adapt-surface'])
 for x in ['gt','surface','fixed-correspondence','source-provenance','method-name','method-commit','out']:p.add_argument('--'+x,required=True)
 a=p.parse_args();require_current_protocol(CURRENT_CONFIG)
 scope=json.loads((O/'ground_truth/scope_lock.json').read_text());assert scope['status']=='PASS'
 assert scope['evaluator_code_sha256']==repair_evaluator_code_hashes()
 assert scope['current_config_sha256']==sha256_file(CURRENT_CONFIG)
 assert Path(a.gt).resolve() in [(O/'ground_truth'/s/'ground_truth/gt.npz').resolve() for s in scope['scope']]
 value=adapt_surface(a.surface,a.gt,CURRENT_CONFIG,a.fixed_correspondence,a.out,method_name=a.method_name,method_commit=a.method_commit,source_provenance=json.loads(Path(a.source_provenance).read_text()))
 print(json.dumps(value['summary']),flush=True)
if __name__=='__main__':main()
