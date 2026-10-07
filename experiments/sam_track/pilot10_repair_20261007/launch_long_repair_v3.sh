set -eu
CODE=/home/chenkejun/CVPR/experiments/pilot10_repair_20261007
OUT=/data/chenkejun/CVPR/revisable_instance_map/pilot10_repair_20261007/long_tracking_20261007/repair_v3_20261007/room2-14047feb0df3aaa5
PY=/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python
mkdir -p "$OUT"
if [ -f "$OUT/v3_summary.json" ]; then echo 'Trial already complete'; exit 0; fi
if [ -f "$OUT/run.pid" ]; then
  EXISTING_PID=$(cat "$OUT/run.pid")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    EXISTING_COMMAND=$(tr '\000' ' ' < "/proc/$EXISTING_PID/cmdline")
    case "$EXISTING_COMMAND" in *"$CODE/run_long_repair_v3.py"*) echo 'Existing trial is running'; exit 0;; esac
  fi
fi
"$PY" -m py_compile "$CODE/run_long_repair_v3.py"
"$PY" -c "import sys; sys.path.insert(0,'$CODE'); import run_long_repair_v3 as x; x.check_batch_equivalence(); print('Batch replacement equivalence and rollback PASS')"
df -h /data/chenkejun/CVPR
free -h
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$PY" -u "$CODE/run_long_repair_v3.py" > "$OUT/run.log" 2>&1 < /dev/null &
echo $! > "$OUT/run.pid"
cat "$OUT/run.pid"
