# Two-audit 第三环节：只读实例不一致性审查 v1

## 已确认范围

本环节按用户确认的三点实现：将稳定簇对应到修复前固定地图 ID；要求两个簇的已归属成员在同帧属于不同原始 Mask，且各自在同一个旧 ID 上至少支持 25 个表面点，这种分离在至少两个独立视角复现；输出疑似过合并及证据，核对现有三状态 ROI 的几何覆盖。没有新增覆盖比例门槛，不生成新 ROI，不触发修复，不修改原地图，不使用 GT。

离线修复和 online-v2 使用相同观测/几何契约；本轮仅运行两个完整离线场景，未接入完整 online-v2 检查点调度。全深度增量 TSDF＋DBSCAN 基线中的最终处理位置不变，审查读取修复前 native 标签，不读取后续修复或终端 DBSCAN 的输出作为自证。

## 输入绑定与实现

- `instance_repair/two_audit_map_inputs.py`：固定地图 ID 与原始 ROI 来源校验。
- `instance_repair/two_audit_anomaly_core.py`：覆盖计数、同帧分离和共同观察视角组。
- `instance_repair/two_audit_anomalies.py`：构建、完整输出及从输入重算验收。
- `instance_repair/tests/test_two_audit_anomalies.py`：16 项数值、来源、时序和端到端测试。
- `scripts/render_two_audit_anomaly_review.py`：服务器端代表证据对照图与静态复核页，不改变判定。

地图固定读取输入缓存所绑定工作区的 `mapping/baseline/native/instance_surface.npz`。NPZ 文件哈希须与第一环节缓存锁完全一致，xyz 的形状、类型、顺序和字节须匹配 geometry_id；禁止替换为同形状的另一份标签或另一个点索引空间。

ROI 固定读取同一 mapping 工作区旁的原始 `candidates/catalog/<scene>`。核对 catalog、membership、verification、mapping_input_freeze，确保其 native 标签、final 标签、三状态证据、图和相机配置来自同一个修复前快照。ROI 的实际帧列表、checkpoint 和来源哈希是绑定依据；历史模板残留的自由文本 method 描述不作为帧范围依据。

ROI 的成员点索引必须有序、唯一、在当前几何范围内；核对核心点哈希、核心/周边点数、三状态计数、核心属于周边范围，以及每个别名对应的组件。ROI 视图和搜索帧不得引用观测缓存之外的帧。源目录和已有输出禁止覆盖。

## 统计口径

逐条处理第二环节 STABLE 簇的已归属原始观测。临时簇和待定观测不参与本轮异常证据。

每个观测输出所有地图标签的点数和占完整 Mask 表面支持的比例，包括未分配标签；另输出该旧 ID 在当前有效可见表面中的点数及覆盖比例。旧 ID 只有大于 0 才可成为异常目标。每个稳定簇再对成员表面支持取唯一点并集，报告各 ID 的唯一点数、占簇支持比例、占该旧实例完整表面比例、出现帧数及视角组数，避免把重复观察次数当作更多独立表面点。

对同一帧的不同稳定簇成员，遍历双方共同覆盖的正旧 ID。双方各至少 25 点时记录一次 JOINT_SEPARATION_SUPPORT，否则记录 INSUFFICIENT_TARGET_SUPPORT，不形成有效分离证据。必须是不同原始 Mask，支持必须属于同帧有效可见集合，两个 Mask 的表面支持必须互斥。不同帧分别出现、待定观测与同一 Mask 重用都不能充当共同观察。

独立视角只统计同一对簇、同一个旧 ID 的有效共同观察帧，不把两个簇各自的视角组数相加。沿用固定视角代表：平移至少 0.1 m 或光轴方向差至少 10° 才能新建视角组，滚转不计为观察方向变化。达到两个组才标 POTENTIAL_OVERMERGE；只有一个组保留为 INSUFFICIENT_INDEPENDENT_VIEWS。不同簇对的视角证据不会汇总凑足条件。

POTENTIAL_OVERMERGE 始终是待审候选，不是已确认错误。候选对按旧 ID 汇总成实例级审查记录，保留全部 pair_key，不重复把同一个旧 ID 计为多个实例。首次证据可用帧仅表示在当前固定几何和标签下何时证据已齐备，不代表另一个在线前缀几何当时已经能检出。

## 三状态 ROI 覆盖的解释

三状态核心来自未分配区域，其周边范围还可能覆盖已分配点，因此不能只按目标旧 ID 与核心点是否相交判定漏检。

对每对候选簇分别报告：

1. 两簇全部已归属成员的完整表面支持并集与 ROI 的交集。
2. 上述并集中属于当前目标旧 ID 的子集与 ROI 的交集。
3. 双方各自完整支持的覆盖，以及具体重叠的 ROI 组件/别名和原选择视图帧。

核心与周边交集分开计数，跨 ROI 使用点集合并集去重，不累加重复成员。完整支持并集与所有 ROI 周边范围交集为零，才记录 `outside_all_selected_roi_context=true`。其他情况保留 CORE_OVERLAP 或 CONTEXT_ONLY 及精确比例，不新增覆盖比例阈值。

“有几何重叠”不等于原流程识别出了这次实例错误，也不等于产生或接受了合适修复种子。“完全无几何重叠”也不等于已经确认发现了真实漏检。真实错误判断和候选种子去向需要后续人工/流程审查。

## 输出与运行

结果根目录：`/data/chenkejun/CVPR/results/two-audit/instance_anomalies_v1_20261104`；各场景位于 `offline/<scene>`。

| 文件 | 内容 |
|---|---|
| observation_instance_coverage.jsonl | 每个稳定成员对所有旧 ID 的支持与覆盖比例 |
| track_instance_coverage.json | 每个稳定簇唯一表面支持上的 ID 对应 |
| stable_track_supports.npz | 同一 geometry_id 下每簇的支持并集 |
| same_frame_pair_checks.jsonl.gz | 全部有共享正 ID 的同帧比较，包含点数不足的记录 |
| candidate_pairs.json | 有至少一次有效同帧证据的簇对及共同观察视角组 |
| instance_anomalies.json | 达到两组条件的疑似过合并旧 ID，按 ID 去重 |
| pair_summary.csv | 逐簇对的帧数、视角组、比例与 ROI 覆盖 |
| review_samples.json | 代表性证据来源，仍为未标注状态 |
| request.json / manifest.json / validation.json | 输入、代码、参数、输出哈希与重算验收 |
| review/index.html | 服务器上的原始 RGB/Mask 代表帧对照页 |
| review/render_validation.json | 图像来源、哈希与格式校验，未声称完成人工视觉标注 |

```bash
cd /home/chenkejun/CVPR/worktrees/two-audit
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python -B \
-m instance_repair.two_audit_anomalies build --tracks STAGE2_OUTPUT --output NEW_STAGE3_OUTPUT

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python -B \
-m instance_repair.two_audit_anomalies verify --audit STAGE3_OUTPUT --verify-inputs

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python -B \
-m scripts.render_two_audit_anomaly_review --audit STAGE3_OUTPUT --output STAGE3_OUTPUT/review
```

静态复核页每个旧 ID 展示共同视角最多的一对簇；若存在完全位于 ROI 周边范围之外的另一对，额外展示该对。每对展示前两个固定共同视角代表。所有其他候选仍在完整 JSON/CSV 中，没有因为代表图选择而被删除。原始 RGB 哈希必须与原始 Mask 推理元数据的 input_sha256 一致，Mask 哈希及尺寸也核对；每帧同时展示全图与候选区域放大图，标出实际旧 ID 点数/支持比例。图片和页面均保留在服务器，未下载到本地。

## 2026-11-04 离线结果

| 场景 | 稳定成员观测 | 同帧旧 ID 比较 | 有效共同观察簇对 | 疑似过合并簇对 | 疑似旧 ID | 完全位于 ROI 范围外的簇对/旧 ID |
|---|---:|---:|---:|---:|---:|---:|
| 5ee7c22ba0 | 4923 | 8597 | 629 | 159 | 29 | 0 / 0 |
| 825d228aec | 3033 | 5154 | 834 | 157 | 17 | 6 / 1 |

其余 470 / 677 对虽有有效同帧证据，但不足两个独立共同视角，继续保留待审，不纳入疑似过合并列表。第一场景的 159 对均与 ROI 核心有几何重叠；第二场景分别为核心重叠 148 对、仅周边重叠 3 对、完全不重叠 6 对。

第二场景完全无 ROI 几何重叠的 6 对全部属于旧 ID 14，由 T001139、T001140、T001143、T001146 四个稳定簇两两组成。各对有 2–4 个共同观察帧、2–3 个独立视角组；这些共同观察中双方支持落入 ID 14 的比例最低为 74%。这是优先人工核查的证据，尚不能排除同一物体不同局部的解释。

不设置覆盖比例门槛会保留大量低比例对应。对每次共同观察先取双方旧 ID 支持比例的较小值，对每对簇跨帧取中位数，再对所有疑似簇对取中位数，结果为 4.22% / 9.88%。这提示必须审查少量交叉标签、边界污染或局部区域造成的候选；本轮未据此自动调参。

构建耗时约 44.13 / 42.52 s，包含输入来源检查和输出，不含之后的完整重算与图像导出。两个任务均使用 CPU，峰值内存约 809 / 747 MiB。16 项测试通过；测试首次因夹具把输出放入受保护的输入工作区而被正确拒绝，隔离测试输入目录后通过，生产算法未因此变更。

每个离线输出都重新从缓存支持、固定标签和 ROI 成员重算并与全部产物比较。另以独立 Python Counter/set 与相机公式核对实际覆盖、支持并集、同帧计数和共同视角，记录在结果根目录 `independent_set_validation.json`。阶段汇总及提交来源记录在 `phase3_validation.json`。

本环节没有真实错误标注、修复收益或 F1/AP50 结果；后续正式开发评价仍必须从规定的 main 工作区使用当前 R2 协议，不能用上述候选数量替代质量评价。
