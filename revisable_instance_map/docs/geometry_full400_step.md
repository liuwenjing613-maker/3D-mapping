# 第 5 步：room0 全场景共享 TSDF 容量验证

固定输入协议：Replica room0，帧 0、5、…、1995，共 400 帧。每次从空图出发，只将 RGB-D、内参和相机位姿用于共享 TSDF 几何积分。输入加载器会读取 mask，但几何模块不使用它。没有实例关联或修复，也没有使用未来帧更新过去帧。

## 实现与推荐参数

入口：`tools/profile_full_geometry.py`。采用 Open3D 0.19.0 CPU、1 cm 体素、8³ 体素/块、深度上限 10 m。针对该 400 帧场景，推荐体素块容量 100,000；实际使用 67,485 块，保留约 32.5% 余量。该容量已设为几何类和完整场景脚本的默认值。10 帧烟测明确指定 20,000 块，保持原实验设置。

运行命令：

```bash
/data/chenkejun/ovimap_runtime_20260908/envs/perception/bin/python \
  /home/chenkejun/CVPR/revisable_instance_map/tools/profile_full_geometry.py \
  --config /home/chenkejun/CVPR/revisable_instance_map/configs/replica_room0_stride5.json \
  --output-dir /data/chenkejun/CVPR/revisable_instance_map/geometry_full400_1cm_capacity100k \
  --reference /home/chenkejun/beauty/ovimap_aligned_eval_20260908/reference/room0/reference.npz \
  --block-count 100000 --checkpoint-every 50
```

## 结果

| 容量 | 实际块数 | 峰值进程内存 | 总耗时 | 表面点数 | 保存的 TSDF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250,000 | 67,485 | 3,571 MiB | 24.5 s | 1,633,034 | 691,858,167 B |
| 100,000 | 67,485 | 2,098 MiB | 23.7 s | 1,633,034 | 691,858,167 B |

推荐运行的 50 帧进度记录见 `progress.json`，详细逐帧与质量数据见 `geometry_full400.json`。其保存结果还包含 `tsdf_grid.npz` 和 `surface.ply`。在独立进程重新加载保存的 TSDF 后，体素块数和提取点数与保存前完全相同。

推荐运行的帧 0、1000、1995 光线投射深度覆盖分别为 99.65%、99.996%、99.995%；与输入深度在共同有效像素上的绝对误差中位数分别为 5.81、4.11、6.41 mm。表面点抽样到参考几何的单向最近距离中位数为 5.50 mm，99.70% 在 5 cm 内。参考几何仅用于运行后检查，不参与建图。光线投射误差是已融合帧的自一致性检查，不是对未见视角的泛化评估。

Open3D 在光线投射时提示范围图分片数不足，并自动扩容后完成计算；三个抽查帧均得到有效结果。容量 100,000 是该房间和该输入协议的实测选择，换场景或分辨率时需重新监控块数与内存。

本步仍不创建实例，也不做关联、冲突检测、提示分割或任何修复。

