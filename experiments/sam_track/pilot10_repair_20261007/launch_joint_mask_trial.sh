set -eu
CODE=/home/chenkejun/CVPR/experiments/pilot10_repair_20261007
OUT=/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/joint_mask_replacement_20261007/room2_mask14_mask24
PY=/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python
mkdir -p "$OUT"
if [ -f "$OUT/v3_summary.json" ]; then echo 'Trial already complete';exit 0;fi
if [ -f "$OUT/run.pid" ]; then
  PID=$(cat "$OUT/run.pid")
  if kill -0 "$PID" 2>/dev/null; then
    COMMAND=$(tr '\000' ' ' < "/proc/$PID/cmdline")
    case "$COMMAND" in *"$CODE/run_joint_mask_trial.py"*) echo 'Existing joint trial is running';exit 0;; esac
  fi
fi
"$PY" "$CODE/joint_vote_ops.py"
"$PY" -m py_compile "$CODE/run_joint_mask_trial.py"
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -u "$CODE/run_joint_mask_trial.py" > "$OUT/run.log" 2>&1 < /dev/null &
echo $! > "$OUT/run.pid"
cat "$OUT/run.pid"
