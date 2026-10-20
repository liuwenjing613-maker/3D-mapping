#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo 'usage: evaluate_ablation.sh NAME ASSOC_DIR MAP_DIR' >&2; exit 2; fi
name="$1"
assoc="$2"
map="$3"
root=/data/chenkejun/CVPR/revisable_instance_map
out="$root/evaluation_${name}"
py=/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python
protocol=/home/chenkejun/CVPR/unified_eval/configs/replica_ca_v1.json
gt=/data/chenkejun/CVPR/evaluation_results/debug_room0/gt.npz
"$py" tools/adapt_instance_map_replica_ca.py \
  --instance-surface "$map/instance_surface.npz" \
  --materialization-report "$map/materialization_report.json" \
  --association-report "$assoc/association_report.json" \
  --gt "$gt" --protocol "$protocol" --output-dir "$out/adapter"
PYTHONPATH=/home/chenkejun/CVPR "$py" -m unified_eval.cli eval-scene \
  --config "$protocol" --gt "$gt" --pred "$out/adapter/canonical_prediction.npz" \
  --out "$out/metrics"
