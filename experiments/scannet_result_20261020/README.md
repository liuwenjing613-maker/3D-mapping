# ScanNet：严格一票无修复、E1 与 auto-v3-swap partial

本分支保存两个既定 ScanNet 场景的实测结果、逐 GT 分析和实现差异。唯一修复收益基线是**严格一票无修复**。历史结果单列，仅供参考。

当前已完成并归档：scene0011_00。

| 场景 | 方法 | AP% | AP50% | F1% | PQ% | TP/FP/FN |
|---|---|---:|---:|---:|---:|---|
| scene0011_00 | 严格一票无修复 | 1.5199 | 8.1884 | 28.1250 | 16.7672 | 9/28/18 |
| scene0011_00 | E1 whole | 0.8787 | 4.7339 | 19.7802 | 11.8818 | 9/55/18 |
| scene0011_00 | auto-v3-swap partial | 0.8787 | 4.7339 | 19.7802 | 11.8820 | 9/55/18 |

## 分析与结果

- [scene0011_00 分析](results/scene0011_00/analysis_scene0011_00.md)：逐 GT、结构诊断、E1/partial 差异及局限。
- [scene0011_00 PLY](results/scene0011_00/ply)、[原生/最终 NPZ 预测](evidence/scene0011_00/prediction_maps)、[完整评分](results/scene0011_00/evaluation_r2/comparison.json)。

[总指标和历史参考](results/comparison_all.json) · [中文对比说明](results/对比说明.txt) · [实现差异说明](IMPLEMENTATION_DIFFS.md)

## 固定评价口径

scene0011_00 为 238 帧 / 27 GT；scene0608_00 为 339 帧 / 34 GT。使用服务器现有步长10序列，保留原始帧号。新三组共用相同 TSDF、关联、GT范围和对应索引。
评分导入 main 工作区锁定的 R2 实现，配合历史已审核的 ScanNet 场景范围适配；不是把 Replica 的350 GT套到ScanNet。评分源码基准 `cd260569e89de1e9525f82c45ee10842b80616b0`，审核归档 `d035f68770cefab27c9751e97062c72faa07367c`，评价 `frozen=false`。main默认CLI当前限定Replica，本次实际入口是归档的 `code/evaluate.py`。
历史 P1-A1 使用反向投影投票；本次严格基线使用 E1 正向投影与同帧同点严格一票，因此跨历史结果的差异不能全部归因于修复。

## 源码与复核

父版本为 auto-v3-swap `1103d8196a980874ef743f572d3b4c0748a4fdab`。`source_inventory.json`逐文件记录来源、SHA-256和继承路径；与父版本任何已有文件字节完全一致的源码不重复新增。有差异或新增的运行文件放在 `code/`，可对应到原源码的文本差异放在 `diffs/`。冻结 E1、auto-v3注册和父版本源码均未修改。
在仓库根执行 `python -B experiments/scannet_result_20261020/verify.py`，核验全部归档哈希、继承源码和指标一致性。它是只读校验，不重跑模型或改写服务器结果。
完整模型重算仍需要服务器原始图像、深度、模型及环境。快照保留原绝对路径用于溯源，切勿直接运行launcher覆盖已完成输出；重跑应使用新的CVPR目录并记录路径适配。
PLY保留instance_id、state和原始surface_index；三状态文件中0为黑、1为黄、3为红。六个NPZ预测分别对应三方法的native/final输出。大规模中间追踪缓存和模型权重通过来源锁定位，未重复上传。
