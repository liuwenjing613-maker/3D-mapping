from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
from official_rule_eval import C,O,S,SCENES,GTROOT,read,sha,dump,evaluate,aggregate
def main():
 assert read(O/'official_rule_adapter_validation.json')['status']=='PASS'
 paths=[C/'official_rule_eval.py',C/'test_official_rule_eval.py',Path(__file__),
  S/'scripts/eval_inst_seg.py',S/'scripts/utils/mesh_postprocess_utils.py']
 paths.extend(GTROOT/s/'gt_instance_mesh.ply' for s in SCENES)
 frozen={str(p):sha(p) for p in paths}
 dump(O/'official_rule_implementation_freeze.json',{'files':frozen,'approval':'user selected OVI evaluation rules primary, R2 auxiliary on 2026-10-31',
  'comparison_kind':'same 400-frame inputs; direct integer IDs; native 5cm projection and original instance score function; no extra VLM/color postprocessing'})
 pending=set(SCENES);done=[]
 while pending:
  for scene in SCENES:
   if scene not in pending:continue
   native=O/'native'/f'{scene}_400_trace';ours=O/'ours'/f'{scene}_400_main'
   log=O/'driver'/(scene+'_0.log')
   if (native/'readout_complete.json').exists() and (ours/'complete.json').exists() and log.exists() and 'NATIVE_DONE' in log.read_text():
    evaluate(scene);pending.remove(scene);done.append(scene)
    dump(O/'official_rule_status.json',{'stage':'RUNNING','completed':done,'pending':sorted(pending)})
  if pending:time.sleep(10)
 result=aggregate()
 for p,h in frozen.items():assert sha(p)==h,('changed_source_or_GT',p)
 dump(O/'official_rule_status.json',{'stage':'COMPLETE','completed':done,'comparison':result})
if __name__=='__main__':
 try:
  dump(O/'official_rule_status.json',{'stage':'STARTING'});main()
 except BaseException as e:
  dump(O/'official_rule_status.json',{'stage':'FAILED','error':repr(e),'traceback':traceback.format_exc()});raise
