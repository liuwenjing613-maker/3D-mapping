# 统一 v3 最新评估：P1-A1、OVI-MAP、OVO、OpenVox

评估日期：2026-10-07；GitHub 整理日期：2026-10-08。六种方法／版本各八个 Replica 场景，共 48 组评估。

**所有结果为 `DEBUG_ONLY / NON_OFFICIAL`。**统一 v3 参数尚未冻结；这些是项目内类别无关指标，不是官方 ScanNet、COCO 或论文表格复现分数。

## 八场景整体结果

单位 %；整体按实例／顶点池化，场景等权均值另见 JSON。

| 方法 | AP | AP50 | AP25 | PQ | F1@50 | mCov | mWCov | AR@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OVI-MAP（400帧） | 3.75 | 9.98 | 36.53 | 23.18 | 31.42 | 38.75 | 42.62 | 16.45 |
| P1-A1 原生 | 5.61 | 16.05 | 48.41 | 28.86 | 39.73 | 41.89 | 43.30 | 18.01 |
| P1-A1 补洞后 | 6.13 | 19.01 | 48.41 | 31.09 | 43.43 | 42.41 | 43.77 | 18.77 |
| P1-A1 自动修复（10月4日） | 6.13 | 18.87 | 48.05 | 30.99 | 43.29 | 42.40 | 43.78 | 18.80 |
| OVO 首公开版（输入预算不同，仅参考） | 3.53 | 10.61 | 32.09 | 22.85 | 32.08 | 42.88 | 44.79 | 18.40 |
| OpenVox 表面适配（输入预算不同，仅参考） | 0.96 | 3.92 | 18.86 | 12.88 | 19.29 | 24.12 | 32.35 | 6.75 |

## 协议与比较边界

- 主映射为 1cm 竞争最近实例；结构诊断使用独立 2cm 几何支持。GT 至少 100 顶点，保留全部预测，包括映射为空的实例。
- 统一置信度、一对一匹配、严格 IoU 阈值及 101 点 AP。八场景共 332 个有效 GT 实例；所有版本共用同一 GT 和有效配置哈希。
- P1-A1、OVI-MAP 每场景使用相同的 400 输入帧。P1-A1 原生／补洞版固定于 10 月 2 日，自动修复版固定于 10 月 4 日。
- OVO 固定首公开提交 `d01d821bec7c25c437803b8ab9c580e73f92e119`，原生输入 2000 帧、配置为 400 次几何更新／200 次分割，SAM2.1 + SigLIP learned。它与主对比的输入及前端预算不同，只作参考。首公开提交不等同于已证明的论文表格私有版本。
- OpenVox TSDF 表面适配使用 2000 几何帧／200 次实例处理，同样只作参考。OVO 与 OpenVox 是不同方法。
- mCov 是所有有效 GT 的最佳 IoU 均值；mWCov 按参考顶点数加权。AR@all 使用全部预测并平均 0.50:0.05:0.95 的召回，不能称为 COCO AR@100。SQ 只对正确匹配实例计算。
- 另保存 P/R/F1@25/50/75、表面覆盖、完整度、纯度、过合并／分裂／重复率，共 19 项扩展指标；没有从最终地图推算时序身份指标。
- 历史 OVO v2 的 AP50=30.49% 使用不同投影、大小过滤及评分口径，不能与本次 v3 混算。

P1-A1 补洞版的 AP50、PQ、F1 高于 OVO；OVO 的 mCov 略高，分裂率为 42.17%，P1-A1 为 23.80%。两者原生预算不同，不能据此声称受控的算法优势。

## 文件与复现

- [comparison.json](comparison.json)：六种方法／版本的完整主指标、19 项扩展指标、pooled／macro 和逐场景数据。
- [per_scene.csv](per_scene.csv)：48 组逐场景结果，数值为 [0,1] 比例。
- [protocol.json](protocol.json)：此次有效 v3 配置。
- [metric_definitions.json](metric_definitions.json)：指标定义和一手来源。
- [verification.json](verification.json)：GT、协议、评估器和原生输入／canonical 文件的哈希与核验记录。原始地图和完整运行档案留在数据盘。

复现脚本保持本次指标计算逻辑，新增环境变量用于调整数据、评估器快照及输出目录。它们重放已固定的实验布局，不负责下载原生地图或重新训练。

```bash
export CVPR_EVALUATOR_ROOT=/path/to/frozen-evaluator-snapshot
export CVPR_DATA_ROOT=/path/to/CVPR-data
export CVPR_EVAL_AUDIT_OUT="$CVPR_DATA_ROOT/results/evaluation_audit_20261007"
export CVPR_OVI_NATIVE_RESULTS=/path/to/ovimap-native/results
export CVPR_OVO_SOURCE_ROOT=/path/to/ovo-first-public-release-run
export CVPR_OVO_AUDIT_OUT="$CVPR_DATA_ROOT/results/ovo_evaluation_audit_20261007"
python scripts/audit_existing_evaluations_20261007.py
python scripts/extend_v3_metrics_20261007.py
python scripts/evaluate_ovo_v3_20261007.py
python scripts/extend_ovo_v3_metrics_20261007.py
```

`CVPR_EXTENDED_METRICS_OUT` 可单独指定扩展指标输出；每次运行前应清除该覆盖项或明确设置。脚本依赖已有实验 manifests 中的源文件路径，迁移数据时须保持路径可访问；它们检查已记录的源哈希，不接受换图或改 GT。

本次使用 P1-A1 固定快照中的 v3 评估器；需与 verification.json 的评估器文件哈希一致，不能用未经验证的其他代码替换。执行需要 NumPy、SciPy，OVO 结果补算不启动 GPU 建图。
