from online_common import *
import os,subprocess,time,traceback
GPU='GPU-b003c3ac-453c-dd7b-90b0-31ad9adb3163'
ENV={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','CUDA_DEVICE_ORDER':'PCI_BUS_ID','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(C/'guard')+':'+str(C)+':/home/chenkejun/CVPR/repos/sam2_1','CVPR_ONLINE_GUARD':str(R/'input_guard.json'),'XDG_CACHE_HOME':str(R/'runtime_cache'),'CUDA_CACHE_PATH':str(R/'cuda_cache'),'LD_LIBRARY_PATH':'/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09','CUDA_VISIBLE_DEVICES':GPU}
def run(name,relative,args=(),python=PY):
 start=time.perf_counter();unix=time.time();p=None;code=None;record={}
 try:
  with (R/(name+'.log')).open('ab') as log:
   p=subprocess.Popen([python,'-B','-u',str(C/relative),*map(str,args)],cwd=(C/relative).parent,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   write(R/'workflow_status.json',dict(arm='B',stage=name,pid=p.pid,unix=time.time()));print('STAGE',name,p.pid,flush=True);code=p.wait()
  if code:raise RuntimeError(name+' failed '+str(code))
 finally:
  record=dict(arm='B',stage=name,seconds=time.perf_counter()-start,started_unix=unix,ended_unix=time.time(),pid=p.pid if p else None,returncode=code,status='PASS' if code==0 else 'FAILED_OR_INTERRUPTED')
  with (R/'process_timing_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
 return record
def main():
 assert read(R/'mapping_complete.json')['frame_count']==100
 run('catalog','catalog/build_candidates.py',['room0'])
 run('prepare','frontend/prepare_inputs.py')
 run('exact_model_cache','reuse_exact_models.py')
 run('cropformer','frontend/infer_cropformer.py',python=CROP)
 run('sam','frontend/infer_sam.py',python=SAM)
 for step in ['summary','overmerge','intersection','versions','semantic_classify','semantic_cabinet','semantic_route','grouping_r2','grouping_r3','v1','semantic_background','v2']:
  if step=='summary':run(step,'frontend/summarize_frontend.py')
  elif step in ['overmerge','intersection','versions','grouping_r2','grouping_r3','v1']:run(step,step+('/score.py' if step=='overmerge' else '/compute.py'))
  elif step=='semantic_classify':run(step,'semantic_route/classify.py')
  elif step=='semantic_cabinet':run(step,'semantic_route/refine_cabinet.py')
  elif step=='semantic_route':run(step,'semantic_route/route.py')
  elif step=='semantic_background':run(step,'v2/semantic/run.py',['--manifest',R/'v1/seed_manifest.json','--raw-config',C/'config.json','--out',R/'v2/semantic','--config',C/'v2/semantic/config.json'])
  elif step=='v2':run(step,'v2/measure_and_select.py')
 write(R/'selection_complete.json',dict(status='PASS',arm='B',candidate_processed_frame_range=[51,100],candidate_raw_frame_range=[250,495],scoring_history=[1,100],repair50_applied=True,repair100_applied=False,GT_used=False))
 run('all_edge_prepare','all_edge/screen.py',['100','prepare'])
 run('all_edge_select','all_edge/run_variant.py',['100',GPU])
 run('all_edge_render','all_edge/render.py',['100'])
 run('qa','qaB.py')
 write(R/'workflow_status.json',dict(status='PASS',arm='B',stage='100_new50_masks_ready_no_second_repair',unix=time.time()))
if __name__=='__main__':
 try:main()
 except Exception:write(R/'workflow_failure.json',dict(error=traceback.format_exc(),unix=time.time()));raise
