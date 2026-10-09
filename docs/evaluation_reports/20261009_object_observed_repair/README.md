# V3 物体观测修复评估：改动与验收（2026-10-09）

已实现独立 `object_observed_repair` profile，并完成八场景 × 四种方法的开发重评。77 项测试、8 项验收及 96 个落盘文件哈希核验通过。当前是 DEVELOPMENT，尚未正式冻结。

代码基于 `origin/main@337d112`，位于服务器任务分支 `codex/v3-object-observed-repair-20261009`；评分运行提交为 `7fbe2d5`。历史 V3 的 332-GT 配置、公式与结果保持可复现；新 profile 的 350-GT 分数不能直接与其比较。

## 改了什么

| 文件 | 具体改变与目的 |
|---|---|
| `gt_scope.py`、`replica.py` | 保留原始 GT 身份，分开 TARGET/NON_TARGET/UNKNOWN/INVALID；取消 100 顶点资格门槛，6 个极小对象单独待审。 |
| `observable_surface.py`、`mesh_visibility.py` | 固定 400 帧 GT 网格遮挡、实例 ID 和有效深度证据；完整四边形网格确定性三角化，无最小像素门槛；固定观测 mask/count 和哈希。 |
| `geometry.py` | 全部 TSDF 几何参与一次最近邻，保留未分配位置；标签版本只读取同一索引；原实例清单和空投影实例保留。 |
| `metrics.py` | TARGET、已知背景、IGNORE 显式分区；有目标支持的未匹配预测计 FP，取消新 profile 的 50% void 豁免；背景/未知/不可验证预测另报。 |
| `repair_profile.py`、P1 适配器 | 接入固定映射和成对校验；OVI 原 ID 0 正确规范化并保留原 ID、类别来源；拒绝几何、帧列表、scope 或配置变化。 |
| `schema.py`、`io.py`、新配置 | 保存原始标签与区域；强制新旧 profile 分离、配置和数组哈希一致；本开发版不能直接翻转 frozen 标志冒充正式结果。 |
| 运行脚本、两个测试文件、协议说明和 README | 统一重评、oracle、扰动与小物体结构审核，说明运行方式与全部限制。最后仅修正日志中的源四边形面数/渲染三角形面数名称，不改变渲染和评分。 |

保留 V3 的匹配、AP/PQ/F1/mCov 计算；新增正确/错误归属、有几何未分配和无几何四项 coverage，以及逐 GT 改善/退化报告。coverage 按参考顶点计，不能解读为平方米面积。

## GT 范围

604 个原始非 void GT 实例：372 个目标候选（366 个来源验证通过、6 个待审）、64 个已知非目标、168 个 UNKNOWN。另有 6 个原始 void ID 记录。366 个通过资格的候选中，16 个没有可信观测，最终可评价对象是 350 个。34 个不足 100 顶点的非极小目标中，31 个有可信观测；其余仍保留身份并说明观测不足。

UNKNOWN 官方来源为 148 个 class_id=-1 和 20 个缺 ID 记录。用户已确认暂无额外映射、继续 UNKNOWN；不当背景、不计普通 FN。此确认未改变评价范围。

| 场景 | 可评价 GT | 有资格但无可信观测的 ID | TSDF 几何 oracle PQ |
|---|---:|---|---:|
| room0 | 72 | 66 | 0.9674 |
| room1 | 45 | 无 | 0.9779 |
| room2 | 52 | 无 | 0.9543 |
| office0 | 31 | 21, 31, 37 | 0.9644 |
| office1 | 23 | 4, 9, 13, 24, 38 | 0.9575 |
| office2 | 41 | 9, 74 | 0.9276 |
| office3 | 58 | 37, 112 | 0.9475 |
| office4 | 28 | 50, 64, 67 | 0.9111 |

## 开发结果

八场景 pooled 统计；四种方法共用相同 GT scope 和观测范围。P1 三版本共用完全相同的 TSDF 几何与对应索引；OVI 使用自己的完整原生几何。

| 方法 | PQ | F1@50 | mCov | 正确归属 coverage |
|---|---:|---:|---:|---:|
| P1 原始 | 0.4784 | 0.6208 | 0.5492 | 0.6792 |
| P1 补洞 | 0.4970 | 0.6354 | 0.5609 | 0.6902 |
| P1 自动修复 v2 | 0.4966 | 0.6345 | 0.5610 | 0.6901 |
| OVI-MAP full400 | 0.4310 | 0.5606 | 0.5475 | 0.7119 |

原始→补洞：248 个 GT best-IoU 提升、69 个下降、33 个不变，正确归属增加 17,778 顶点。补洞→自动修复：93 个提升、48 个下降、209 个不变，但正确归属减少 115 顶点。自动修复不能被笼统宣称为全面改善；IoU 增减与正确表面增量必须同时报告。

## 八项验收

| 检查 | 证据与结果 |
|---|---|
| room1 小目标不被尺寸删除 | 4 个插座 ID 14/16/44/49、开关 ID 31 均参与评价；可信观测分别为 70/77/70/77/60 顶点，覆盖各自全部参考顶点。通过。 |
| UNKNOWN 与背景/FN 分开 | 168 个原始 UNKNOWN 保留，全部不可评价；另有显式 FP/ignore 反例测试。通过。 |
| 修复几何与对应完全相同 | 8 场景逐一检查 P1 三版本 xyz、缓存文件、索引哈希；落盘文件再核验。通过。 |
| 删除错误标签不能增加正确归属 | 删除反例测试通过；报告同时记录未分配和正确顶点增量。通过。 |
| 人为合并/分裂反映退化 | 质量和结构反例通过；32 次重评均含主结构与小目标敏感性表。通过。 |
| 空投影重复实例不能逃避 FP | 重复与真实空 inventory 反例通过；实际导出清单全部保留，无 dropped UID。通过。 |
| 重命名 ID 不改变质量 | PQ/F1/AP/mCov/coverage/结构反例通过。通过。 |
| 离散完美 GT 与几何 oracle 分开 | 8 场景离散 AP50/F1/PQ/mCov/正确归属均为 1；几何 oracle PQ 为 0.9111–0.9779，残差已记录。通过。 |

另已输出预先固定的 1/3/5/10/20 mm 位移审核、主结构 10 顶点且 5% 与辅助 1 顶点且 5% 的对照。它们是冻结前审核资料，不能据方法排名选择参数。

## 仍待确定

6 个极小对象：room0:76（vase，4 顶点）、office0:43（lamp，4）、office1:10（blanket，4）、office3:20（table，14）、office3:35（window，6）、office3:109（window，4）。保留 TARGET 候选、待审核，不自动判无效。

几何映射 1 cm、结构支持 2 cm、观测深度差 2 cm 和结构显著性门槛仍是开发参数；需审核观测/几何 oracle 残差与小目标稳定性后另行冻结。开发实现、反例验收和八场景重评已完成，正式冻结仍待上述审核。

## 文件与复现

代码：`/home/chenkejun/CVPR/worktrees/v3-object-observed-repair-20261009`。完整结果：`/data/chenkejun/CVPR/results/v3_object_observed_repair_20261009`。所有原生地图保持原文件。

机器证据：`acceptance_checklist.json`、`artifact_integrity_audit.json`、`policy_decisions.json`、`unified_eval_tests.xml`、`mesh_acceptance_report.json`、`mesh_comparison.development.csv` 和逐场景 adapter manifest。协议说明：`docs/OBJECT_OBSERVED_REPAIR_V3_CN.md`。
