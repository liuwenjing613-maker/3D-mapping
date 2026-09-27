# 统一评估器 v2：指标与计算协议

## 状态与范围

`Replica-CA-v1` 原样保留，用于复现旧结果。`Replica-CA-v2` 使用同一 Replica 八场景 GT、参考网格、实例标注和输入帧清单。v2 的每实例投影改变了预测掩码，因此**必须从原生地图重新适配，不能把 v1 的结果与 v2 的结果混列**。

v2 继承投影距离 `δ=0.05 m`、有效 GT 实例至少 `100` 个参考顶点和未匹配预测的忽略比例门槛 `>0.5`。显著交集阈值 `N` 与 GT 占比阈值 `α` 尚未校准：配置中为 `null`，运行时必须显式指定临时值，所有输出标为 `DEBUG_ONLY / NON_OFFICIAL`。校准后才可冻结配置并做正式比较。ScanNet 官方 AP 由独立的官方评估器输出，不与这里的 `CA-AP_uniform` 同列。

## 0. 统一几何与忽略规则

每个预测实例**独立**投影：对每个参考顶点，找该实例点云的最近点；距离严格 `<δ` 时纳入该实例。两个预测可以覆盖同一顶点；即使投影后为空，也保留该原生实例。只有预测掩码确实互不重叠时，`is_partition=true`。输出含每实例掩码和预测间重叠顶点数。该步骤不改变 GT 网格及其标签。

ConceptGraphs 从原始 `pcd_*.pkl.gz` 适配；OVI-MAP 从原生网格导出的 `xyz/instance/native_instance_ids` 适配，其中 `instance` 是实例表序号，映射回原生 ID 后再逐实例投影。两者使用同一参考网格、距离门槛、实例匹配和指标计算。输入帧清单及原始文件哈希写入各自 manifest。

有效 GT 实例按当前评估表面上的顶点数过滤，至少 `100` 个。预测实例**没有最小尺寸过滤**。过小 GT 的顶点从预测的 IoU 面积中排除，并与背景、stuff 一起计入忽略区域。未匹配预测若原始有效掩码中忽略区域占比 `>0.5`，不计 FP；等于 `0.5` 仍计 FP。空投影且无忽略支持的原生实例计 FP。在线前缀中，GT 尺寸按该前缀已观测顶点重新判断。

## 1. 最终地图实例质量

| 指标 | 计算 | 解释 |
|---|---|---|
| `CA_AP50_uniform`（主） | 所有预测同分，IoU 严格 `>0.50`；同分组整体做最大数量优先、IoU 次优先的一对一匹配；由该组的 P/R 做 101 点插值 AP | 同分顺序不影响结果；只有一个置信度工作点，**不是官方 AP** |
| `CA_AP25_uniform`、`CA_AP_uniform` | 前者 IoU `>0.25`；后者对 `0.50,0.55,…,0.95` 十档 AP 求均值 | 补充空间精度诊断 |
| `CA_PRF1_0_5` | 用 IoU `>0.50` 的同一一对一匹配：`P=TP/(TP+FP)`，`R=TP/(TP+FN)`，`F1=2TP/(2TP+FP+FN)` | 重叠预测也能算；无有效 GT 时记为未定义 |
| `CA_PQ` | 只在预测为无重叠划分时计算。`SQ=TP 的平均 IoU`，`RQ=TP/(TP+0.5FP+0.5FN)`，`PQ=SQ×RQ` | `RQ` 数值等于同口径 F1；有重叠时 PQ/SQ/RQ 为 N/A，不强行改写预测 |

所有 TP/FP/FN 都按场景独立匹配；未匹配且忽略区域占比 `>0.5` 的预测不计 FP。AP 采用统一同分组协议，和官方 evaluator 的排序、类别及匹配规则分开。

## 2. 错误结构与表面诊断

先定义预测 `p` 与有效 GT `g` 的**显著交集**：

`|p∩g| ≥ N` **且** `|p∩g|/|g| ≥ α`。

仅用投影后非空、忽略占比不超过 `0.5` 的预测计算下表的结构比例；`N, α` 来自同一 v2 配置。

| 指标 | 分子 / 分母 | 说明 |
|---|---|---|
| `split_gt_rate` | 显著连接至少 2 个预测且未归入 duplicate 的 GT 数 / 有效 GT 数 | 检查一个真实实例被拆成多个预测 |
| `merge_prediction_rate` | 显著连接至少 2 个 GT 的预测数 / 合格预测数 | 检查一个预测吞并多个真实实例 |
| `duplicate_prediction_rate` | 未匹配、且对已被另一预测匹配的 GT 有 IoU `>0.5` 的预测数 / 合格预测数 | 先识别重复实例；对应 GT 不再重复计入 split |
| `macro_best_gt_recall` | 每个 GT 对所有预测的最大 `|p∩g|/|g|`，再对 GT 平均 | 完整性；漏掉的 GT 以 0 参与平均 |
| `macro_prediction_purity` | 每个合格预测对所有 GT 的最大 `|p∩g|/|p|`，再对预测平均 | 纯净度；不能单独证明没有 split |
| `gt_surface_coverage` | 被任一预测覆盖的有效 GT 顶点数 / 有效 GT 顶点数 | 判断质量变化是否来自删掉难区域；不判断身份是否正确 |

显著 merge/split/duplicate 是本项目的诊断指标，不是官方 benchmark 指标。分母为 0 时比例记为 `null`，不会伪装成 0。原始交集、IoU、precision、recall 仍保存在 `overlap_matrix.npz`。

## 3. 跨场景与在线过程

批量输出同时保留两种汇总：顶层数值是**pooled**（合并各场景 TP/FP/FN 或预测与 GT 后计算），`macro_per_scene` 是各场景指标的**不加权平均**。二者分开报告，避免大场景掩盖小场景。PQ 若任一场景因重叠不可计算，整体 PQ 也为 N/A；不能只平均可算的场景。

`eval-online-prefix` 按原实验 manifest 的帧清单顺序读取深度、内参和位姿。从第 1 帧累计到 checkpoint `t`，仅在已观测参考顶点上评估**该时刻已提交的地图快照**，输出 `CA_AP50_uniform(t)`、`P/R/F1(t)`、`PQ(t)` 和 merge/split/duplicate 比例。未观测 GT 表面不参与该前缀。快照必须声明 `committed_frame_id=t` 且 `max_input_frame_id≤t`；评估器拒绝声明使用未来帧的快照。元数据检查是因果性验收的一部分，不能替代对建图程序是否实际读取未来数据的审计。

在线 manifest 示例（路径相对该 manifest；实际 `frames` 必须完整覆盖源实验的原有帧列表）：

```json
{
  "scene_id": "room0",
  "source_experiment_manifest": "experiment_manifest.json",
  "source_method": "my_method",
  "frames": [
    {"frame_id": 0, "depth_m": "depth_0.npy", "intrinsics": "K.npy", "world_from_camera": "pose_0.npy"}
  ],
  "checkpoints": [
    {"frame_id": 0, "prediction": "map_at_0.npz"}
  ]
}
```

## 本版边界

Repair Success、False Repair、Repair Delay 尚无冻结事件定义，本版不实现；`repair.py` 保持原样。ScanNet200/ScanNet++ 官方适配和验收不在本版改动内。v2 的显著交集阈值未冻结，尚无可用于论文的 v2 八场景结果。
