set -eu
CODE=/home/chenkejun/CVPR/experiments/pilot10_repair_20261007
OUT=/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/joint_mask_replacement_20261007/room2_mask14_mask24/tracking
PY=/data/chenkejun/CVPR/runtime/sam2_1_env/bin/python
mkdir -p "$OUT"
if [ -f "$OUT/complete.json" ]; then echo 'Joint tracking already complete'; exit 0; fi
if [ -f "$OUT/run.pid" ]; then
  PID=$(cat "$OUT/run.pid")
  if kill -0 "$PID" 2>/dev/null; then
    COMMAND=$(tr '\000' ' ' < "/proc/$PID/cmdline")
    case "$COMMAND" in *"$CODE/track_joint_chairs.py"*) echo 'Joint tracking already running';exit 0;; esac
  fi
fi
GPU_UUID=$(nvidia-smi --query-gpu=uuid,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2>=14000 && $3<10 {print $1;exit}')
if [ -z "$GPU_UUID" ]; then echo 'No suitable idle GPU'; exit 2; fi
"$PY" -m py_compile "$CODE/track_joint_chairs.py"
nohup env CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -u "$CODE/track_joint_chairs.py" > "$OUT/run.log" 2>&1 < /dev/null &
echo $! > "$OUT/run.pid"
echo "Joint tracking launched on $GPU_UUID"
cat "$OUT/run.pid"
