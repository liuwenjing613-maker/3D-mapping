# P0.1：Surface-to-frame projective evidence

## 范围

P0.1 只替换实例证据投影方向。CropFormer/深度前端、11,729 条观测、persistent instance 关联和 400 帧共享 TSDF 均保持不变。建图程序不读取 GT。P0.2 的 tentative 发布、冲突求解、3×3 fallback 和表面邻域补全均未加入。

每个 TSDF 表面点投影到每一帧，先检查视锥和有效深度，再要求投影深度与 RGB-D 深度差不超过固定容差，随后读取该像素的 frame-local mask ID 并映射到 persistent instance ID。因此每帧对每个表面点天然只有零或一个实例证据。逐帧 `(surface point, local mask ID)` 在应用 persistent ID 前单独保存，以支持后续重新关联。

## 验证

新增实现：

- `src/revisable_instance_map/surface_projective_evidence.py`
- `tools/materialize_surface_projective_evidence.py`
- `tests/test_surface_projective_evidence.py`

11 项单元测试全部通过，包括旋转和平移位姿逆变换、视锥裁剪、深度遮挡门控、背景 mask 排除及每帧单标签不变量。两组完整运行均为 400 帧、1,633,034 个 TSDF 点，并与 P0 的输入哈希完全一致。产物检查确认逐点总票数、状态、主图标签和逐帧清单一致；两份 PLY 均包含 1,633,034 点，坐标最大误差为 0。

## room0 结果

主图仍只发布 `CONFIRMED`，保证这里只比较证据投影，不混入 P0.2。

| 版本 | Confirmed | Tentative | Conflict | Unsupported | AP50 | PQ | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 pixel→surface，1.5 cm | 1,436,742 | 16,925 | 72,216 | 107,151 | **0.4300** | **0.4768** | 38 / 11 / 30 |
| P0.1 surface→frame，1.5 cm | 1,471,097 | 5,695 | 65,520 | 90,722 | 0.3951 | 0.4600 | 37 / 14 / 31 |
| P0.1 surface→frame，2.0 cm | 1,477,082 | 5,933 | 65,802 | 84,217 | 0.3951 | 0.4612 | 37 / 14 / 31 |

P0.1 达成了结构目标：1.5 cm 下 confirmed 增加 34,355 点，unsupported 减少 16,429 点，conflict 减少 6,696 点，同帧多 ID 点从 47,742 降为 0。2.0 cm 继续提高覆盖，但多出 282 个 conflict。

统一指标说明当前单像素查询不是最终版本。P0.1 的 SQ 从 P0 的 0.7340 提高到 0.7400（2.0 cm 为 0.7442），说明匹配成功实例的表面质量更好；RQ 从 0.6496 降到 0.6218，导致 AP50/PQ 下降。P0 与 P0.1 都为 confirmed 的 1,424,202 个点中，99.906% 的实例标签一致，指标变化由少量边界和小实例的关键点变化放大。

结果目录：

- `/data/chenkejun/CVPR/revisable_instance_map/surface_p01_room0_depth15mm/`
- `/data/chenkejun/CVPR/revisable_instance_map/surface_p01_room0_depth20mm/`
- `/data/chenkejun/CVPR/revisable_instance_map/evaluation_surface_p01_room0_depth15mm/`
- `/data/chenkejun/CVPR/revisable_instance_map/evaluation_surface_p01_room0_depth20mm/`
