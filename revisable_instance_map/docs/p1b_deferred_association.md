# P1-B：风险观测隔离与逐观测延迟提交

基线为 P1-A `55c099c38a34752c0daa5d7f3a430f120f4001b1`，独立分支 `codex/p1b-deferred-association`。
验收合同保存在 `p1b_implementation_spec_cn.md`；固定开发参数为 `configs/p1b_development_defaults.json`。
参数沿用合同起点，没有逐场景搜索，不声称最优。

## 状态与计票

`AssignmentLedger.raw_support_store` 保留所有不可变观测、真实源帧、源 mask 哈希、深度/位姿/内参/mask 内容哈希、采样配置及 3 cm 支持。
`assignment_store` 单独保存 ACCEPTED、PENDING_BIND、PENDING_BIRTH、UNPROJECTABLE、正式 ID、归属版本和证据时间。
原 `FrameIdentityEvidence` 只接受 ACCEPTED 的真实源帧贡献，同帧/同体素/同实例一票。暂存索引不参与其概率分母。

`Q...` 是审核工作集 ID。它不会进入正式实例命名空间，也不证明同包观测是同一个物体。
同一工作集可部分放行，混合成员继续暂定；超出活动窗口休眠，新的空间命中可重新唤醒，不因超时强制建立实例。

## 快速关联、审核与冷启动

可靠旧实例须通过继承的绝对资格门槛、0.05 分差、至少 3 个可见支撑、原 0.2 可见重叠率，以及局部命中中至少一半有两帧以上支持。
候选先评分 top8。未评分且原始覆盖率达 0.25 的候选，用未舍入覆盖率上界 `0.5*G_binary+0.5` 检查是否可能影响分差；必要时扩展至预算 32。无法排除遗漏候选时仍暂定。
相对 softmax 权重只记录解释，不是可靠概率证书，不乘入票数。

逐观测审核从原始 mask 像素、深度及位姿恢复实际世界点，进行双向 mask/深度一致性验证，不拿格子中心充当多视图证据。
无共享可见支撑是 UNKNOWN；明显跨多个独立原始区域保留 MULTI_REGION_CONTRADICTION ticket。
快速路径可用同一冻结旧地图独立确认的相同 ID 解释同帧碎片；暂定审核仍保守要求原始区域相容。

旧 ID 放行要求至少两个有新增信息的真实帧锚点，同意一个旧 ID。每个锚点必须在排除目标 Q 已放行来源后独立可靠匹配该旧 ID。
排除按原有引用数重建局部计票，同帧其他来源的票仍保留。
原始同包视图可以作为几何见证；Q 的暂定身份和已写入的结论不能作为身份依据。

空地图使用新生路径：至少两个原始相容、非近重复的实际视图，候选检索完整，且没有通过继承资格门槛的旧解释。
这是对合同“没有可靠旧解释”的保守落实；它会保留部分存在弱旧候选的观测，避免以新 ID 绕过模糊旧关联。
新生 cohort 只合并逐张通过原始核验的观测，记录 seed 和正式 ID，不能把整个 Q 批量搬入新实例。

近重复按位移 <2 cm、转角 <5 度且新增 sampled voxel 比例 <0.1 合并；同一源帧永远只有一组。
分组用于减少明显重复，不声称统计独立。没有独立替代证据时保留失败，不虚构自愈。

## 同帧事务与因果时间

当前全部评分、原始审核锚点都读取同一个 `read_map_version`。本帧计划中的放行和新生 ID 不能互相成为正式证明。
提交前校验来源、状态、归属版本、真实见证帧、支持哈希和时间顺序；激活、账本、暂存/空间索引、生命周期和日志作为同一事务。
激活后、账本后、空间索引后、最终记账后四个阶段故障都须撤回整批并恢复版本、高水位和队列。

迟到的放行只激活源帧贡献。例如源帧 100 的观测在 115 确定身份，计票源帧仍是 100；决定帧为 115。
所有实际核验过的原始视图均纳入最大证据帧；不能仅报告最终正见证的帧号。
`decisions_at_arrival.jsonl` 永久保留初始判断，`associations.jsonl` 是最终归属；`decision_events.jsonl` 可重建指定决定时刻的归属。
已 ACCEPTED 的历史观测不会随 argmax 自动重派，完整因果依赖修复留待后续层。

每帧最多 32 个工作集、每包最多 16 条观测和 32 个见证，观察目标按轮转续排；预算未处理的目标保留队列。
只缓存相同冻结版本/相同排除集合的评分、不可变支持包围盒和已到达的位姿，减少重复计算，不增加确认票。

## 表面发布与恢复

`rebuild_p1b_surface.py` 校验原 P0 来源目录、配置、manifest、逐帧区域哈希、关联表与 P1-B 检查点。
主输出使用 `top1_votes/(accepted_total+unresolved_count_upper) >= 0.67` 且至少两票。
`u` 按不同 observation/表面点计上界，同帧不同 mask 不合并；不进入身份证据概率场。
`p0_accepted_only.npz` 仅用于诊断。pending-only 区域有独立理由码；阻塞区输出 `diffusion_protected_mask`。
本版主实验不启用扩散；未来适配器必须保留保护标记，不能利用未知区补全绕过身份审核。

最终 TSDF 只用于完整输出后的离线重发布和固定 v3 评估，不用于在线关联、见证或早期质量主张。
P1-B 独立检查点保存全量来源、当前归属、事件、Q/cohort、见证、队列、游标、参数及全部高水位，派生正式和暂存缓存重建校验。
A2 导入接口保留接受状态、原修订记录及退休 ID 高水位；缺少决定帧的历史修订只在导入时刻生效，不捏造早期知识。主 B2 实验始终从空图开始。

## 对照与命令

| 模式 | 行为 |
|---|---|
| A2 | 原 P1-A，从空图硬关联 |
| B0 | 新来源/归属接口，所有 A2 判断直接接受，必须逐字段/逐数组等价 |
| B1 | 风险隔离和新生确认，关闭模糊旧 ID 的延迟绑定 |
| B2 | 完整隔离、原始独立见证和逐观测放行 |

服务器运行：

```bash
cd /home/chenkejun/CVPR/worktrees/p1b-deferred-association
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python
$PY -m unittest discover -s revisable_instance_map/tests -v
$PY revisable_instance_map/tools/audit_deferred_cases.py --out /data/chenkejun/CVPR/revisable_instance_map/RUN/controlled_cases.json
$PY revisable_instance_map/tools/run_p1b_validation.py --scenes room0 room1 room2 office0 office1 office2 office3 office4 \
  --frame-count 400 --jobs 8 --audit-prefix \
  --parameters revisable_instance_map/configs/p1b_development_defaults.json \
  --output-root /data/chenkejun/CVPR/revisable_instance_map/RUN
```

`run_p1b_validation.py` 从空图分别运行 A2/B0/B1/B2，重建原生表面，用原统一 Replica-CA-v3 DEBUG_ONLY / NON_OFFICIAL 口径评价，不修改评估器。
它校验 B0 等价、完整来源重建、检查点恢复、独立 50 帧与完整运行第 50 步封存归属/计票一致，以及运行中的源码/参数冻结。
另报发布覆盖、小 GT 召回、残留 pending、延迟、逐观测原始见证和离线 GT 诊断。
GT 诊断采用区域占优和剔除整个源帧的身份参照；不是官方关联真值，未能评估的观测单独报告。
消融 `expand_candidates=false`、`exclude_self_support=false` 可由独立参数文件运行；accepted-only 表面必须单列，不能取代主输出。

具体结果和适用边界见后续验收报告，不能仅凭测试通过宣称优于 A2。
