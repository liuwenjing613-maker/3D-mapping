set -e
H=/home/chenkejun/CVPR/experiments/p1a1_roi_cropformer_zoom_20261003
R=/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003
mkdir -p "$R/cuda_cache" "$R/runtime_cache"
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 CUDA_CACHE_PATH="$R/cuda_cache" XDG_CACHE_HOME="$R/runtime_cache" LD_LIBRARY_PATH=/data/chenkejun/CVPR/libs/p1a1_roi_cropformer_zoom_20261003/nvidia_580_126_09 /data/chenkejun/ovimap_runtime_20260908/envs/cropformer/bin/python -u "$H/infer_crops.py" > "$R/inference_native.log" 2>&1 < /dev/null &
echo "NATIVE_INFERENCE_PID=$!"
