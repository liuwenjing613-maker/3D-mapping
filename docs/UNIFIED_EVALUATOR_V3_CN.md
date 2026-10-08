# Replica-CA-v3：实现与校验（2026-09-27）

## 结论

已实现 v3 **实验版**，旧 v1/v2 和官方 evaluator 未改。v3 的竞争映射保留全部原始实例，并把主指标与结构诊断分开；合成测试通过，但八场景 GT oracle **未通过 PQ=1 的冻结门槛**。因此配置仍为 `frozen: false`，所有下述数值仅用于协议校准，不得进入论文正式比较表。

## 评估器的四层

| 层 | 输入与计算 | 输出 |
|---|---|---|
| 1. 统一参考 | 继承现有八场景 GT mesh、实例标注、有效/忽略顶点、原实验帧列表与坐标变换 | `CanonicalGT`；不重新采样 |
| 2. 主几何映射 | 每个参考顶点找最近的**原始预测实例**；最近距离 `< δ_geo` 才归属，且只归一个实例。全部原始实例保留，零顶点实例也保留 | partition `CanonicalPrediction`；零顶点实例参与 FP |
| 3. 主质量 | 主预测与 GT 的交集/并集构成 IoU；IoU `>0.5` 最大一对一匹配得到 P/R/F1；同分预测成组匹配并做 101 点插值计算 `CA-AP50_uniform`、`CA-AP25_uniform`、`CA-AP_uniform`；`PQ=SQ×RQ`，`SQ=匹配IoU均值`，`RQ=TP/(TP+0.5FP+0.5FN)` | AP/F1/PQ、completeness、purity；与官方/native AP 分列 |
| 4. 错误结构 | 各原始实例独立以 `δ_diag` 支持 GT 顶点，得到 `S_ij=被实例i支持的GT-j顶点数/GT-j有效顶点数`；达到显著交集数和比例才连边，统计 merge/split；重复实例用诊断 IoU 与一对一匹配判定 | `pairwise_support_matrix.npz`、merge/split/duplicate；**不进入主 AP/PQ** |

`δ_geo`、`δ_diag`、GT 最小顶点数、显著交集数与比例仍是待定项。待定配置缺值即报错；命令行显式临时值一律标为 `DEBUG_ONLY / NON_OFFICIAL`。v3 的在线前缀评估要求每个 checkpoint 同时提交主预测和诊断预测，且独立声明深度观测容差。

## 实现要点

- `geometry.py`：`map_instances_to_reference_v3()` 竞争式映射；`pairwise_geometry_support()` 只为结构诊断做独立支持。精确坐标并列按点云内容排序，防止文件顺序、实例 ID 改变主指标。
- ConceptGraphs 与 OVI-MAP 适配器同时输出 `canonical_prediction.npz` 和 `diagnostic_support_prediction.npz`。两者保留相同的原始实例顺序，包括空实例；评估入口校验角色、版本和源地图哈希。
- `eval-scene` 分别保存主 `overlap_matrix.npz` 与 `pairwise_support_matrix.npz`；批量评估要求每个场景给出两份预测。正式 `summary.csv` 在待定协议下不生成。
- `eval-official-native` 保持独立，`CA-AP_uniform` 不冒充官方 AP。

## 服务器校验结果

服务器代码：`/home/chenkejun/CVPR/3D-mapping`；数据与完整逐条件结果：`/data/chenkejun/CVPR/evaluation_results/replica_v3_geometry_20260927`。复用八场景原 reference，调试用 GT 最小顶点数 100。服务器完整回归：**45 passed**（`REPLICA_REFERENCE_ROOT=... python -m pytest -q unified_eval/tests`），包括在线前缀的双预测与观测门控校验。

| GT oracle 条件 | 1 cm | 2 cm | 5 cm |
|---|---:|---:|---:|
| 八场景 TP / 332 | 332 | 332 | 322 |
| 八场景 PQ 场景均值 | 0.9972 | 0.9446 | 0.8632 |
| 跨 GT 实例顶点误归属数 | 23 | 23 | 23 |
| room0 25% 顶点 TP / 68 | 0 | 68 | 67 |
| room0 12.5% 顶点 TP / 68 | 0 | 67 | 68 |
| room0 高斯扰动 σ=10 mm 的 TP / 68 | 58 | 68 | 67 |

八场景精确 oracle 在 1 cm 下 AP50/F1 全满分，但 PQ 均未达到 1。room0 的精确 oracle 仍有 7 个本属其他 GT 实例的顶点误归属、507 个背景顶点被覆盖、7 个本实例顶点丢失。原因包含**不同 GT 标签共享同一坐标**造成的几何歧义，以及距离门控仍会覆盖附近背景。对同坐标点，仅凭 native 几何不可能判别其 GT 实例标签。`δ_geo` 缩小可减少背景覆盖，却会让稀疏地图大面积失配；2 cm 能抗稀疏和 10 mm 扰动，但精确 GT 自身 PQ 已降到约 0.9446。

room0 三份可信原始地图各随机抽取 20 万点，对现有参考表面的最近距离 p95 分别为：ali-dev **8.31 mm**，GT mask 0.5 **8.29 mm**，full GT mask **8.36 mm**。这只测点到表面的几何误差；它不能替代 GT 表面到稀疏地图的覆盖测试，更不能按方法排名选择 δ。

合成测试验证：两份近重复预测产生 1 TP、1 FP（另有一个空实例再加 1 FP）；merge/split 被独立诊断识别；多次打乱预测顺序、实例/点顺序，主指标与结构结果不变；改变 `δ_diag` 不改变主 AP/F1/PQ。另一个反例显示：**邻居预测缺席**且两物体间距小于 `δ_geo` 时，唯一归属仍把缺席物体表面分给现有预测，导致该预测 IoU 恶化。竞争映射解决重复占用，但没有完全解决邻物体串扰。

真实 ali-dev room0 地图的端到端调试也通过：适配器保留 72/72 个原始实例，生成主预测和诊断预测；`eval-scene` 生成两份矩阵及 manifest。临时 `δ_geo=δ_diag=2 cm`、GT 最小 100、显著交集 10 顶点且 GT 比例 0.05 时，结果为 TP=22、FP=32、FN=46、F1=0.3607、CA-AP50_uniform=0.1331、PQ=0.2592，**只证明流程能运行，不作为算法排名或参数选择依据**。

## 冻结判定

建议中的硬门槛是精确 GT `TP=332、FP=FN=0、AP50=F1=PQ=1`，同时稀疏/轻微扰动稳定、相邻物体不串扰。当前 v3 **未达到 PQ=1、相邻缺失预测安全、密度稳定三项的共同要求**，不能冻结 `δ_geo`，也不能将此版作为论文正式 Replica-CA-v3 分数。下一步应先明确如何处理参考 mesh 中的重复坐标与附近背景，并在不使用 GT 实例标签修正预测的前提下重新验证映射；之后再按顺序确定 GT 最小尺寸与结构诊断阈值。

## 2026-10-07 补算结果与扩展指标

已核验 P1-A1、OVI-MAP、OVO 和 OpenVox 六种方法／版本共 48 组场景评估，补充 19 项覆盖、召回和结构指标。完整数值、逐场景数据、定义和核验见[最新评估报告](evaluation_reports/20261007/README.md)。OVO、OpenVox 的原生输入预算与主比较不同，作为参考；全部保持 `DEBUG_ONLY / NON_OFFICIAL` 标记。
