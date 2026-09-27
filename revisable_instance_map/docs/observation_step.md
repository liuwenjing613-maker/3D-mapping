# 第 6 步：逐帧原始实例观测

本步只把固定协议中的 400 帧原始 mask 转为可追溯的实例提案目录。共享 TSDF 不随 mask 改变；观测之间尚无跨帧身份或对象融合。

## 数据接口

代码：`src/revisable_instance_map/observations.py`。入口：`tools/build_raw_observations.py`。

每条 `RawInstanceObservation` 含以下字段：

- `observation_id`：`Replica/room0/f{帧号}/m{帧内mask编号}`，在固定输入协议内稳定。
- `frame_id`、`mask_local_id`、`source_mask_sha256`：回到只读原始 mask 的来源。
- `pixel_count`、`projectable_pixel_count`：分别统计全部 mask 像素，以及有限且深度在 (0, 10 m) 的像素。
- `bbox_xyxy_exclusive`：精确二维外接框，右下坐标不包含。
- `world_centroid_m` 与 `world_aabb_min_m/max_m`：可投影像素的世界坐标统计。若没有可投影深度，则为 null。

`source_pixel_indices(frame, observation)` 可从原始 mask 精确恢复该提案的全部行优先像素索引，包括无效深度像素。这些索引可在后续被细分；原始 mask 编号不是最终实例 ID，也不是不可拆的最小单元。当前目录只保存摘要和来源，不复制 RGB-D、mask 或逐像素三维点。后续读取时仍应核对来源文件的 SHA-256。

## 运行与结果

```bash
/data/chenkejun/ovimap_runtime_20260908/envs/perception/bin/python \
  /home/chenkejun/CVPR/revisable_instance_map/tools/build_raw_observations.py \
  --config /home/chenkejun/CVPR/revisable_instance_map/configs/replica_room0_stride5.json \
  --output-dir /data/chenkejun/CVPR/revisable_instance_map/raw_observations_room0_stride5
```

结果：`observations.jsonl`、`observation_manifest.json`、`progress.json`。共 400 帧、12,046 条观测；每帧 15–46 条，中位数 30。原始 mask 提案覆盖 324,967,642 个像素，其中 324,963,048 个具有可投影深度。所有非零编号均保留：1,274 条观测少于 100 像素，最小 1 像素；没有观测完全缺少可投影深度。摘要目录约 6.3 MB，建表耗时 42.7 秒，峰值进程内存约 183 MiB。

逐帧检查通过：mask 文件哈希匹配现有元数据；观测数匹配前端记录；观测像素和可投影像素的总和分别严格等于源图统计；观测 ID 无重复。对帧 0、1000、1995 共 84 条观测做了全部源像素回溯。另抽查 9 条观测，与既有独立投影函数计算的世界坐标中心一致，最大差异小于 0.1 微米。

这些数据是未经筛选的二维提案。大型 mask 的三维外接框可能同时跨多个物体，不能当成可靠对象边界。此步没有 GT、跨帧关联、实例状态、分割修订或提示分割器。
