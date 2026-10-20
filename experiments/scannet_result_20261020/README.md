# ScanNet：严格一票无修复、E1 与 auto-v3-swap partial

本分支保存两个既定 ScanNet 场景的实测结果、逐 GT 分析和实现差异。唯一修复收益基线是**严格一票无修复**。历史结果单列，仅供参考。

当前已完成并归档：scene0011_00, scene0608_00。

| 场景 | 方法 | AP% | AP50% | F1% | PQ% | TP/FP/FN |
|---|---|---:|---:|---:|---:|---|
| scene0011_00 | 严格一票无修复 | 1.5199 | 8.1884 | 28.1250 | 16.7672 | 9/28/18 |
| scene0011_00 | E1 whole | 0.8787 | 4.7339 | 19.7802 | 11.8818 | 9/55/18 |
| scene0011_00 | auto-v3-swap partial | 0.8787 | 4.7339 | 19.7802 | 11.8820 | 9/55/18 |
| scene0608_00 | 严格一票无修复 | 3.5015 | 15.1660 | 35.2941 | 22.3488 | 21/64/13 |
| scene0608_00 | E1 whole | 3.3279 | 13.1542 | 31.8182 | 20.3681 | 21/77/13 |
| scene0608_00 | auto-v3-swap partial | 3.3279 | 13.1542 | 31.8182 | 20.3756 | 21/77/13 |

当前已完成场景中，E1与partial的F1、PQ均低于严格一票无修复基线；partial和E1整体接近。以下计数来自固定IoU 0.5评分，不能只按局部IoU改善个数判断总体收益。

- scene0011_00：TP 9→9，FP 28→55；E1相对严格基线F1 -8.3448、PQ -4.8854个百分点。
- scene0608_00：TP 21→21，FP 64→77；E1相对严格基线F1 -3.4759、PQ -1.9807个百分点。

## 分析与结果

- [scene0011_00 分析](results/scene0011_00/analysis_scene0011_00.md)：逐 GT、结构诊断、E1/partial 差异及局限。
- [scene0011_00 PLY](results/scene0011_00/ply)、[原生/最终 NPZ 预测](evidence/scene0011_00/prediction_maps)、[完整评分](results/scene0011_00/evaluation_r2/comparison.json)。
- [scene0608_00 分析](results/scene0608_00/analysis_scene0608_00.md)：逐 GT、结构诊断、E1/partial 差异及局限。
- [scene0608_00 PLY](results/scene0608_00/ply)、[原生/最终 NPZ 预测](evidence/scene0608_00/prediction_maps)、[完整评分](results/scene0608_00/evaluation_r2/comparison.json)。

[总指标和历史参考](results/comparison_all.json) · [中文对比说明](results/对比说明.txt) · [实现差异说明](IMPLEMENTATION_DIFFS.md)

## GT 实例 PLY

补充两场景本次评价所用 GT 实例的点云，保持原始 ScanNet 坐标。完整表面版展示这 27/34 个目标的全部参考网格顶点；可见区域版严格对应 R2 的评价点集。完整表面仅供查看形状，不扩大评价范围。

| 场景 | 完整 GT 表面 | 评价可见区域 | 配色依据 |
|---|---|---|---|
| scene0011_00 | [113,836 点 / 27 实例](results/scene0011_00/gt_ply/scene0011_00_gt_instances.ply) | [78,514 点](results/scene0011_00/gt_ply/scene0011_00_gt_instances_observed.ply) | [22 个基线色、3 个 E1 色、2 个独立新颜色](results/scene0011_00/gt_ply/README.md) |
| scene0608_00 | [75,865 点 / 34 实例](results/scene0608_00/gt_ply/scene0608_00_gt_instances.ply) | [54,224 点](results/scene0608_00/gt_ply/scene0608_00_gt_instances_observed.ply) | [33 个基线色、1 个 E1 色](results/scene0608_00/gt_ply/README.md) |

颜色优先沿用严格一票无修复结果，再使用 E1/partial 的已有颜色。采用一对一重叠匹配，防止预测合并时把两个 GT 涂成同色。scene0011_00 的 GT 20、27 使用独立颜色；两场景分别有 8、2 个弱重叠颜色对应，详见各目录的 `color_mapping.json`。这些是可视化对应，不代表预测正确，原预测 PLY 和评分不变。

PLY 的 `instance_id` 保留 GT ID（官方 objectId+1），借色来源另存 `color_prediction_id`；保留原始顶点索引和评价区域标记。新文件哈希和验证记录见 [GT 交付清单](results/gt_ply_delivery_manifest.json)，服务器导出实现见 [export_gt_ply.py](tools/export_gt_ply.py)。

## 建图与 GT 的坐标、窗口深度诊断

[完整诊断说明](diagnostics/geometry_gt_20261020/错位与反光诊断.txt) · [统计与 PLY 坐标核验](diagnostics/geometry_gt_20261020/diagnosis_summary.json)

两场景三方法的固定 XYZ 完全一致，577 帧适配深度和位姿逐一核对通过。窗口范围内深度缺失约 69.7% / 88.0%，但这些数据不能单独证明反光是主要原因；帧间重投影同时证实局部观测残差。GT 实例 PLY 只含评价目标，预测 PLY 还含墙面/地面，需避免把显示范围差异当成整体错位。所有配准与窗口平面仅用于诊断，未改动几何、mask、人工标注或评分。

- scene0011_00：[同坐标对照](diagnostics/geometry_gt_20261020/scene0011_00/same_coordinates_comparison.png)、[窗口原始深度](diagnostics/geometry_gt_20261020/scene0011_00/window_aperture_diagnostic.png)、[逐帧深度对照](diagnostics/geometry_gt_20261020/scene0011_00/raw_depth_examples.png)。
- scene0608_00：[同坐标对照](diagnostics/geometry_gt_20261020/scene0608_00/same_coordinates_comparison.png)、[窗口原始深度](diagnostics/geometry_gt_20261020/scene0608_00/window_aperture_diagnostic.png)、[逐帧深度对照](diagnostics/geometry_gt_20261020/scene0608_00/raw_depth_examples.png)。

## 固定评价口径

scene0011_00 为 238 帧 / 27 GT；scene0608_00 为 339 帧 / 34 GT。使用服务器现有步长10序列，保留原始帧号。新三组共用相同 TSDF、关联、GT范围和对应索引。
评分导入 main 工作区锁定的 R2 实现，配合历史已审核的 ScanNet 场景范围适配；不是把 Replica 的350 GT套到ScanNet。评分源码基准 `cd260569e89de1e9525f82c45ee10842b80616b0`，审核归档 `d035f68770cefab27c9751e97062c72faa07367c`，评价 `frozen=false`。main默认CLI当前限定Replica，本次实际入口是归档的 `code/evaluate.py`。
历史 P1-A1 使用反向投影投票；本次严格基线使用 E1 正向投影与同帧同点严格一票，因此跨历史结果的差异不能全部归因于修复。

## 源码与复核

父版本为 auto-v3-swap `1103d8196a980874ef743f572d3b4c0748a4fdab`。`source_inventory.json`逐文件记录来源、SHA-256和继承路径；与父版本任何已有文件字节完全一致的源码不重复新增。有差异或新增的运行文件放在 `code/`，可对应到原源码的文本差异放在 `diffs/`。冻结 E1、auto-v3注册和父版本源码均未修改。
在仓库根执行 `python -B experiments/scannet_result_20261020/verify.py`，核验全部归档哈希、继承源码和指标一致性。它是只读校验，不重跑模型或改写服务器结果。
`external_dependencies_inventory.json`补充核对运行来源锁引用的外部第一方依赖与main R2评分文件；已有相同字节仍只引用父版本。
重做分析图表：`python experiments/scannet_result_20261020/tools/analyze_scene.py --results experiments/scannet_result_20261020/results --scene scene0011_00 --output /tmp/scannet-analysis`；第二场景使用`analyze_scene0608.py`并将scene设为`scene0608_00`。需要numpy与matplotlib。
完整模型重算仍需要服务器原始图像、深度、模型及环境。快照保留原绝对路径用于溯源，切勿直接运行launcher覆盖已完成输出；重跑应使用新的CVPR目录并记录路径适配。
PLY保留instance_id、state和原始surface_index；三状态文件中0为黑、1为黄、3为红。每场景六个NPZ预测分别对应三方法的native/final输出。大规模中间追踪缓存和模型权重通过来源锁定位，未重复上传。
