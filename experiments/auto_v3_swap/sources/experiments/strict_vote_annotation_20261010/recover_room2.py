import os,time,json,subprocess
from pathlib import Path
R=Path("/data/chenkejun/CVPR/results/strict_vote_annotation_20261010")
C=Path("/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010/room2")
while not (R/"room0/sam2_1/audit.json").exists():time.sleep(5)
assert json.loads((R/"room0/sam2_1/audit.json").read_text())["status"]=="PASS"
env={**os.environ,"CUDA_VISIBLE_DEVICES":"1","OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1","CUDA_CACHE_PATH":str(R/"room2/cuda_cache"),"XDG_CACHE_HOME":str(R/"room2/runtime_cache"),"PYTHONPATH":"/home/chenkejun/CVPR/repos/sam2_1","LD_LIBRARY_PATH":"/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09"}
with (R/"room2/sam2_1.log").open("ab") as f:subprocess.run(["/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python","-u",str(C/"infer_sam.py")],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
while not (R/"room2/cropformer/audit.json").exists():time.sleep(5)
assert json.loads((R/"room2/cropformer/audit.json").read_text())["status"]=="PASS"
with (R/"room2/render.log").open("ab") as f:subprocess.run(["/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python","-u",str(C/"render_annotation.py")],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
(R/"room2/pipeline_complete.json").write_text(json.dumps({"status":"READY","recovered_gpu":"3 -> 1","parameters_changed":False}))
