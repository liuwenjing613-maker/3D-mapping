# V3 object_observed_repair 开发 profile

本 profile 用于固定 RGB-D 输入下的物体实例建图和标签修复评估。沿用 V3 的一对一匹配、AP、PQ 和 F1 公式，单独修改 GT 资格、可信观测表面、完整几何对应和 unmatched prediction 判定。

配置：`unified_eval/configs/replica_ca_v3.object_observed_repair.development.json`。协议名称仍是 `Replica-CA-v3`，同时必须记录 `evaluation_profile=object_observed_repair`、配置哈希和 GT scope 哈希。历史配置默认 `evaluation_profile=legacy`，原 332-GT 数据和结果没有被替换。两种 profile 不能混合评分。

当前版本全部输出 `DEVELOPMENT / NON_OFFICIAL / PROFILE_NOT_FROZEN`。GT 审核和几何/结构阈值尚未冻结，不能通过把 JSON 的 `frozen` 改成 true 生成正式结果。

## GT 身份和资格

`replica.load_scope_source()` 校验原参考、网格和原始标签文件哈希，保留原始物理对象 ID 和语义信息。历史参考中的 `raw_instance` 实际是 `original_semantic_id*1000+raw_object_id`，新加载器显式解码，不把未知对象合并为背景。

`gt_scope.build_gt_scope()` 为每个原始 ID 生成清单。类别和审核状态来自 GT，不能来自预测或修复效果。

| 对象状态 | 当前规则 |
|---|---|
| TARGET | 沿用历史 V3 的物体类别；插座、开关等小目标不再受 100 顶点限制 |
| NON_TARGET | wall、floor、ceiling；不作为物体 GT，可信可观测表面仍参与预测越界统计 |
| UNKNOWN | 官方类别未定义、来源缺失或不能解明；保留原始身份 |
| INVALID | 原始 void ID；极小对象不能仅因尺寸小而自动置 INVALID |

6 个指定极小对象保留 TARGET 候选身份并设 `quality_verified=false`、`PENDING_GEOMETRY_REVIEW`：room0:76，office0:43，office1:10，office3:20/35/109。当前其他 `quality_verified` 表示标签来源验证、类别一致且没有待审标记，并不声称进行了逐个物体的人工审核。

资格和观测分开：只有 TARGET、质量通过且有可信输入观测的对象进入 TP/FN 分母；IoU 仅使用其可信观测顶点。一个已知物体完全不可观测时保留对象记录，不当作普通 FN。

## 固定可信观测

`observable_surface.py` 检查参考顶点投影、图像边界、正深度、输入有效深度、GT 实例 ID 一致和深度差。每个顶点每帧至多增加一次观测。没有 100 像素或最小物体尺寸门槛。

`mesh_visibility.GTMeshVisibility` 对参考顶点实际查询的像素投射相机射线，最近 GT 网格命中提供遮挡深度和原始实例 ID，等价于查询这些像素的 z-buffer。完整网格的非目标和 UNKNOWN 面同样遮挡后方物体。Replica 原网格是四边形面，明确采用 0–2 对角线拆为两个三角形，不改原文件；源四边形任一顶点标签不同，两个子面都不给出可靠实例 ID。射线方向 camera-z=1，不归一化，命中参数与输入 PNG 的 camera-z 深度一致。

输入深度须同时接近投影顶点和 GT 渲染深度。紧邻插座的墙面像素不能作为插座观测。完全重合坐标却有不同 GT ID 的参考顶点另外标记为几何歧义。

完整输出 `gt_observed_support.npz`（mask/count/ambiguity）、`gt_scope.json` 和 `gt.npz`。每帧 RGB、深度、位姿、内参、原网格、帧列表及观测代码均有哈希。缓存 2D GT 路径只标记 `DEVELOPMENT_CACHE_VISIBILITY_NOT_MESH_VERIFIED`；实际网格路径另标 `DEPTH_AND_MESH_VERIFIED`，整个协议仍为开发状态。

## 固定几何和实例清单

`geometry.build_fixed_surface_correspondence()` 让全部物理表面点参加 reference→native 最近邻，包括 0、负 ID、UNASSIGNED/CONFLICT/INSUFFICIENT。严格距离条件是 `< δ_geo`。索引、距离、原生 xyz 的 dtype/shape/顺序/数值哈希及参考哈希一起保存。

标签版本只通过相同索引读取 ID；P1-A1 原生 ID 必须 >0，其他状态没有有效实例。原始几何、点数、顺序或坐标不同都会拒绝复用缓存及成对评分。建立对应不读取 GT 实例标签。

原生正 ID 清单全部保留，零参考顶点预测不删除。若导出有明确 inventory，会保留其中的空实例；P1-A1 现有 NPZ 没有独立 inventory 时只能从导出的正标签恢复 ID，报告明确写 `positive_labels_in_export`，不能声称恢复了未导出的空假设，也不把内部已注销 ID 强行导出。

OVI-MAP 导出使用 compact index，源 ID 0 合法。适配器将 compact index+1 转为正规范标签，保留 `ovi-id:<原始ID>`、原 ID、规范正 ID 和原模型类别。任何非目标类别信息都不得由 GT 为预测补齐；当前无法确认物体/结构语义的预测类型统一是 unknown。

## IoU 和 FP/ignore

参考区域固定为 TARGET=1、KNOWN_NON_TARGET=2、IGNORE=0。交并比包含 TARGET 和已知非目标表面；UNKNOWN、不可观测、不可信与几何歧义区域不加入交并比。主质量映射仍为 partition；独立几何支持只用于结构诊断。

| 未匹配预测的证据 | 处理 |
|---|---|
| 任一 TARGET 参考顶点支持 | FP；即使其余 80% 在 UNKNOWN 中也不忽略 |
| 只有已知非目标参考支持 | 无可靠类别时 ignore 并报告 background-only；原生明确声明 object 则 FP |
| 无已知参考支持但原始几何靠近 TARGET | FP；用于防止零投影重复实例逃避惩罚 |
| 原始导出 inventory 中确实没有几何的空实例 | FP |
| 仅 UNKNOWN/不可观测支持 | ignore 并报告原因 |
| 完全超出固定参考几何范围 | unverifiable 并单独报告；不能算作正确预测 |

原始几何的区域距离检查仅决定 unmatched prediction 的处理，不修正实例归属。已投影的背景实例不会仅因邻近插座就变为 FP。报告同时给出预测在 IGNORE 区域的参考顶点比例，零投影时该比例为空。

新 profile 不使用旧 `void_fraction>0.5` 判定。历史 profile 的这条规则和原公式保留。

## 修复统计与 oracle

主报告含 PQ、F1@50、mCov、merge/split/duplicate 和背景/未知/不可验证预测数。TARGET 顶点分为正确归属、错误归属、有几何未分配和 NO_GEOMETRY，四个比例和为 1。这里的 coverage 权重是固定参考顶点数，不等同于平方米面积。

正确归属采用最大交集的一对一 GT–prediction 对齐，仅用于评分，不送回算法。删除标签使交集矩阵逐项不增，其最优总正确顶点数不能增加；因此丢弃错误预测不能被宣称为增加了正确归属表面。成对报告的 corrected/degraded instances 明确定义为同一 GT 的 best-IoU 增减，须和正确归属顶点增量一起看，不能把 IoU 上升直接说成新增正确表面。

区分两个 oracle：直接在离散参考使用完美 GT mask 必须得到 AP50/F1/PQ/mCov=1；GT→真实 TSDF→参考的几何 oracle 单独记录残差，GT 派生的测试标签只存在于 oracle，不写入任何方法地图。另以预先给定的 1/3/5/10/20 mm x 方向偏移检查几何门槛敏感性，不根据方法排名选择阈值。

主结构诊断暂保留 10 顶点且覆盖 GT 5%；另输出 1 顶点且 5% 的小物体敏感性表，不能混为主分数。δ_geo=1 cm、δ_diag=2 cm、观测深度差=2 cm 都是开发参数。

## 运行与来源校验

服务器代码所在 task worktree：`/home/chenkejun/CVPR/worktrees/v3-object-observed-repair-20261009`。结果放 `/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009`，不重写任何原地图。

```bash
# 先生成八场景候选清单，并验收 room0、room1
python scripts/run_object_observed_repair_development.py --scenes room0 room1 --visibility mesh

# 两场景验收通过后补完八场景；复用时逐个核验源 RGB/深度文件
python scripts/run_object_observed_repair_development.py \
  --scenes room0 room1 room2 office0 office1 office2 office3 office4 \
  --visibility mesh --reuse-observed

# 对已经生成的固定 GT scope 适配后续标签版本
python -m unified_eval.repair_profile adapt \
  --surface /path/to/instance_surface.npz --gt /path/to/gt.npz \
  --config unified_eval/configs/replica_ca_v3.object_observed_repair.development.json \
  --correspondence /path/to/fixed_gt_to_tsdf.npz --output-dir /path/to/new_adapter \
  --method-name method --method-commit commit --source-provenance /path/to/input_provenance.json

python -m unified_eval.repair_profile paired \
  --before /path/to/before/canonical_prediction.npz --after /path/to/after/canonical_prediction.npz \
  --gt /path/to/gt.npz --config unified_eval/configs/replica_ca_v3.object_observed_repair.development.json \
  --output /path/to/paired.json
```

P1-A1 输入证据来自已校验的 association/materialization 报告及 400 帧输入配置；修复版本还必须通过相同完整几何哈希。OVI-MAP 使用既有逐帧 pose/intrinsic/decoded-RGB 对齐审核并核验当前导出及 manifest 哈希。独立 adapter 未提供方法输入证明时显式标 `method_input_scope_verified=false`，不能将这个标志当作默认通过。

输出有完整 main/diagnostic prediction、交集矩阵、逐实例 ignore 原因、源码与输入哈希、成对改善/退化表、小目标诊断、oracle 及 `*.development.csv/json`。最终在线前缀不能使用这份最终 400 帧 scope；更换输入帧必须单独生成 scope，禁止临时传入 observed_mask 改变分母。

正式冻结还需：6 个极小对象的审核结论、未知 GT 的来源处置确认、观测及离散几何残差审核、小物体结构门槛与几何稳定性验收。不能把开发候选数 366 当作最终可评价 GT 数。
