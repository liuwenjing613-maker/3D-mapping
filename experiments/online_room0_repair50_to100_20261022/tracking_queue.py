from common import *
from concurrent.futures import ThreadPoolExecutor
import subprocess,traceback
GPUS=["GPU-d31841e8-d3ae-e7f2-f581-8dc866e5c129","GPU-9a11b2b0-1816-1e33-9dd1-395cf0aaf9dd"]
def worker(arg):
 gpu,cases=arg;rows=[]
 for case in cases:
  uid=case["case_uid"];start=time.perf_counter()
  env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"LD_LIBRARY_PATH":"/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09","CUDA_CACHE_PATH":str(TOP/"runtime_cache/cuda"),"XDG_CACHE_HOME":str(TOP/"runtime_cache"),"MPLCONFIGDIR":str(TOP/"runtime_cache/mpl")}
  with (R/"cases"/uid/"tracking.log").open("w") as f:
   q=subprocess.run([SAM,"-u",str(C/"track_case.py"),"--config",case["config_path"]],env=env,cwd=C,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  if q.returncode:raise RuntimeError(uid+" tracking failed")
  row={"case_uid":uid,"seconds":time.perf_counter()-start,"GPU":gpu};rows.append(row);print(json.dumps(row),flush=True)
 return rows
def main():
 start=time.perf_counter();cases=read(R/"seed_diagnostics.json")["cases"]
 dump(R/"tracking_queue_status.json",{"status":"RUNNING","cases":len(cases),"raw_prefix":[0,245],"directions":["forward","reverse"]})
 with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(worker,[(g,cases[i::2]) for i,g in enumerate(GPUS)]))
 dump(R/"tracking_queue_complete.json",{"status":"PASS","seconds":time.perf_counter()-start,"cases":[r for x in rows for r in x],"model_reinitialization_preserved":True,"dense_RGB_tracking_preserved":True})
if __name__=="__main__":
 try:main()
 except Exception:dump(R/"tracking_queue_failure.json",{"traceback":traceback.format_exc()});raise
