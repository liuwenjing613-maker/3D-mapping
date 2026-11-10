"""Background display work only; never launches mapping, selection, tracking or repair."""
from pathlib import Path
import json,subprocess,time,traceback
from refresh_progress import refresh,dump,read,ROOT,OUT
from refresh_timing import main as refresh_timing
C=Path(__file__).resolve().parent
PY='/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python'
def passed(suffix):
 return all((q:=read(ROOT/str(n)/g/suffix)) is not None and q.get('status')=='PASS' for n in [200,300] for g in ['A','B'])
def invoke(flag,args):
 flagpath=OUT/flag
 if flagpath.exists():return
 dump(flagpath,dict(status='RUNNING',started_unix=time.time(),argv=args,scope='visualization_only'))
 start=time.perf_counter()
 with (OUT/'display_export.log').open('a') as f:
  result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 dump(flagpath,dict(status='PASS' if result.returncode==0 else 'FAILED',seconds=time.perf_counter()-start,
  ended_unix=time.time(),returncode=result.returncode,scope='visualization_only'))
def main():
 while True:
  try:
   refresh();refresh_timing()
   if passed('repair/repair_complete.json') and not (OUT/'export_ready.json').exists():
    invoke('auto_export_attempt.json',[PY,str(C/'export_compare.py')])
   q=read(OUT/'export_ready.json');m=read(OUT/'manifest.json')
   if q and q.get('status')=='PASS' and (not m or m.get('status')!='PASS') and passed('repair/commit_complete.json') and passed('final_audit.json'):
    invoke('auto_publish_attempt.json',[PY,str(C/'export_compare.py'),'--publish'])
   refresh();refresh_timing();m=read(OUT/'manifest.json')
   if m and m.get('status')=='PASS':
    dump(OUT/'display_background_complete.json',dict(status='PASS',ended_unix=time.time(),scope='all eight PLY plus commit and independent audit gates'));break
  except Exception:
   dump(OUT/'display_background_error.json',dict(status='FAILED',unix=time.time(),traceback=traceback.format_exc()))
   traceback.print_exc()
  time.sleep(20)
if __name__=='__main__':main()
