# OVI-MAP 逐帧深度几何细化接入（Replica room0）

## 范围和依据

这是建图前端的**逐帧** CropFormer + 深度几何细化，不是后续仅在冲突区域按需运行的提示分割器。仅复现 OVI-MAP 前端，不把现有跨帧关联误称为 OVI-MAP 关联器。

对照源码：`OVI-MAP/OVI-MAP` 修订 `f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`，`scripts/utils/common_scannet_nyu.py` 中 `SegmentsGenerator.frameToSegmentsCropFormer`，以及其 `depth_segmentation_py` C++ 扩展。原 CropFormer 缓存由官方 `CropFormer_hornet_3x_03823a.pth`、配置 `cropformer_hornet_3x.yaml`、默认置信度阈值 0.5 生成；分数排序和 mask 覆盖次序与官方 `demo_from_dirs.py` 相同。旧输入仅读。

## 实现

1. `tools/build_ovimap_depth_masks.py` 在原 OVI native 容器内调用**已编译的官方** `depth_segmentation_py.DepthSegmentation_py`。输入 RGB 转 RGB float32、深度 PNG 除以 6553.5、内参为 float32 的 680×1200 相机矩阵。官方模块计算深度图、法线、深度不连续性、最大距离、最小凸性、边缘及几何区域。按官方 `save2DGeometricSegs` 的次序写 uint8 区域 ID。输出在 `/data/chenkejun/CVPR/revisable_instance_map/ovimap_depth_room0_stride5/`。
2. `src/revisable_instance_map/ovimap_refinement.py` 是 MaskFusion 的逐条件 Python 移植：几何区域小于 100 像素舍弃；候选与 CropFormer 区域交叠大于其面积的 0.9、且小于当前几何剩余面积的 0.5 时先拆出交集；剩余区域若最佳交叠不少于剩余面积的 0.2 则按该 CropFormer ID 标记，否则为背景。阈值使用官方严格 `>` / 非严格 `>=` 的边界。背景区域不作为正实例观测。
3. `tools/build_ovimap_refined_masks.py` 逐帧验证 RGB、深度、原 mask、几何 mask 的 SHA-256；写 16 位细化 mask、`frames.jsonl` 和 `segments.jsonl`。后者保留几何区域 ID、CropFormer 来源 ID、拆分/主区域分支、重叠率、像素数和各输入哈希。原 mask 不修改。
4. 细化观测在 `Replica/room0/ovimap-refined-v1/...` 命名空间下产生，与原始 CropFormer 观测 ID 不冲突。`frame_io.py` 支持 uint8/uint16 mask；`build_raw_observations.py` 仍逐区域记录精确像素来源和 3D 统计。用于后续基线的配置：`configs/replica_room0_stride5_ovimap_refined.json`。
5. 共用已有的同 400 帧、同 1 cm TSDF 表面。当前基线关联和表面投票未被改造成 OVI-MAP 的空间投票/合并器，因此下表只是**前端替换实验**。可选 `--allow-multiple-observations-per-instance-per-frame` 仅用于诊断一物多片兼容性，默认行为保持原基线。

## 核验

- 400 张重新生成的几何 mask 与先前 OVI-MAP 原生运行缓存逐像素一致；`frames.jsonl` 每帧记录 `reference_cache_equal=true` 和输入/输出 SHA-256。
- `tools/check_ovimap_fusion_parity.py` 对照官方 `frameToSegmentsCropFormer` 的每帧正区域次序、来源 ID、像素数、重叠率以及背景区域数；结果见 `official_fusion_parity.json`。
- 16,138 条细化区域与 16,138 条观测逐一对应，像素数/有效深度像素数一致；与 12,046 条原始观测没有 ID 碰撞。GT 从未参与分割或关联，只在已完成地图的离线评估中加载。
- 上游 pybind 深度模块在 Python 正常退出时有析构崩溃；生成脚本在所有文件关闭并 flush 后用 `os._exit` 避开。生成结果逐帧与原生缓存相等，未以该绕行改变分割算法。

## 400 帧结果

- 原 CropFormer 12,046 区域；深度几何 mask 13,616 区域；融合后 16,138 个正实例区域，其中拆出区域 2,537 个。
- 原正 mask 像素 324,967,642；细化正 mask 像素 316,213,917。原正像素被剔除 9,895,513，原背景被几何区域吸收 1,141,788。400 帧都有拆分；每帧细化区域数 19–67，中位数 38。
- 2,703 个逐帧 CropFormer 来源 ID 对应不止一个细化区域；最大一个对应 15 个区域。这直接影响“每个已有实例每帧只收一个区域”的基线约束。

| 输入/关联 | 3D 实例 ID | TSDF 表面有标签比例 | CA-AP50 | CA-PQ | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| 原 CropFormer + 原基线 | 863 | 94.05% | 0.2068 | 0.3234 | 34 / 49 / 34 |
| OVI 前端 + 原基线 | 718 | 92.16% | 0.0629 | 0.1676 | 27 / 143 / 41 |
| OVI 前端 + 每帧多区域诊断 | 304 | 94.30% | 0.1266 | 0.2620 | 28 / 64 / 40 |

评估协议均为同一冻结的 `Replica-CA-v1`，为自定义类别无关指标，不是 ScanNet 官方 AP。原基线的 `CA-AP50` 最高；不能把前端替换实验的下降解释为 OVI-MAP 完整方法的效果。多区域诊断缓解了当前关联限制，但未恢复原水平，可能仍有跨区域误合并、细化区域归属或实例票选问题，需要在后续关联设计中独立处理。

## 输出位置

- 代码、配置、文档：`/home/chenkejun/CVPR/revisable_instance_map/`
- 几何 mask：`/data/chenkejun/CVPR/revisable_instance_map/ovimap_depth_room0_stride5/`
- 细化 mask 与逐区域来源：`/data/chenkejun/CVPR/revisable_instance_map/ovimap_refined_room0_stride5/`
- 细化观测：`/data/chenkejun/CVPR/revisable_instance_map/ovimap_refined_observations_room0_stride5/`
- 原约束对照：`association_full400_ovimap_refined/`、`instance_map_full400_ovimap_refined/`、`evaluation_room0_ovimap_refined/`
- 多区域诊断：对应目录名后缀 `_multi`。

只在服务器 CVPR 目录写入新代码和结果；本地没有同步。