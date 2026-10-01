# P1-B 实现规范：隔离风险观测，用新增多视图证据延迟提交身份

## 0. 范围、基线与证据状态

审阅基线：`liuwenjing613-maker/3D-mapping`，分支 `codex/p1a-probabilistic-association`，提交 `55c099c38a34752c0daa5d7f3a430f120f4001b1`。

本文件是代码审阅后的工程设计与验收合同，不是已经集成、运行的 P1-B 实现，也不是性能提升保证。没有修改远端分支，没有重新运行服务器的八场景实验。

审阅了 `association.py`、`identity_evidence.py`、`run_baseline_association.py`、`rebuild_p0_from_regions.py`、P1-A 计票测试、实现文档和验收报告。独立执行了一个小型审计：对 `identity_evidence.py` 的本地副本校验 Git blob SHA，完全对应 `b7e2ec66a2f216b667bf7c3115de4f82ea64d626`；运行 8 项源码核心检查和 4 项设计反例检查，共 12 项通过。它们不是仓库全部 50 项测试，更不是八场景重跑。

最终建议：保留 P1-A 的评分与计票底座；新增“观测来源表—归属状态表—正式证据缓存”分离，风险观测暂存，利用不依赖该暂存假设的新增原始观测做逐观测验证，满足条件再事务式放行。P1-B 不负责自动修订已确认的历史错误；该能力留给后续历史修复层，但共用事务与证据接口。

## 1. P1-A 结果告诉我们什么

以下数值来自该分支 `docs/p1a_validation_20261001.md`，不是此次独立复测。

| 输出 | 原二值 A0/A1 | 概率 A2 |
|---|---:|---:|
| 原生 P0 AP50 | 9.0519% | 10.4016% |
| 原生 P0 F1 | 29.6651% | 32.1608% |
| 原生 P0 PQ | 21.9185% | 24.1386% |
| 同原有扩散后 AP50 | 10.4730% | 12.5783% |
| 同原有扩散后 F1 | 32.1656% | 35.3923% |
| 原生显著过拆分 GT 数 | 68 | 71 |
| 原生显著过合并预测数 | 65 | 64 |
| 原生诊断预测总数 | 298 | 270 |

统一 Replica-CA-v3 仍为当前记录的 `DEBUG_ONLY / NON_OFFICIAL`；不改变评估器版本，不把实验口径写成官方分数。

结论：概率支持值得保留，但结构错误未全面改善。P1-B 不能只靠拒绝更多观测获得漂亮的已发布区域；必须证明暂定观测能够正确回收、覆盖率和小物体召回没有被系统性牺牲。

## 2. 已有代码可以保留的部分

1. `FrameIdentityEvidence` 的同帧引用计数：`R(frame, voxel, instance)`。
2. `C(voxel, instance)` 的不同帧支持计数与经验支持比例。
3. `reassign_observations()` 的显式撤票/加票与归属版本。
4. 持久 ID 高水位：最高 ID 退休后不复用。
5. 整帧先读取旧状态评分，之后统一提交。
6. 3 cm 身份缓存与原生 P0 表面输出分离。
7. `candidate_support_traces` 的最大概率来源格子、支持帧数、总票数记录。
8. 来源哈希、检查点、从账本重建缓存、原生 P0 再发布。

P1-B 不得将这些能力重新实现成另一套互不一致的计票体系。

## 3. 接入 P1-B 前必须修复的接口假设

这些是 P1-A 在“全部观测都有正整数归属”前提下的边界，不应笼统称为现有算法 bug。

### 3.1 登记来源与激活身份证据目前绑在一起

`FrameIdentityEvidence.add(entry)` 登记 entry 后立即 `_add_votes(entry)`。在 P1-A 中 `observation_support` 直接引用 `identity_evidence.entries`。

因此，给一条观测写上 `status=pending`，再照常调用 `_commit()`，不构成隔离；给它一个临时正整数 ID 后调用 `_commit()`，同样会使它进入候选和计票。

P1-B 必须分开：

- `raw_support_store`：所有已接收观测，包含 pending 和 unprojectable；不可变。
- `assignment_store`：当前归属与状态；版本化。
- `active_identity_evidence`：只由 ACCEPTED 观测派生的正式支持。
- `pending_index`：暂定观测的空间检索缓存；只用于找后续关联线索。

### 3.2 当前 q 不是“可靠关联证书”

`_score_candidates()` 对空间召回的至多 8 个候选和 null 做 softmax；普通 `candidates` 仅返回前三。其 q 没有经过真值正确率校准，且计算范围不等于“通过全部资格门控的候选”。

一帧纯支持可使某候选分数达到 1；当 null=0.35、T=0.1 且只有一个候选时，q≈0.9985。这不能证明它是正确实例。

P1-B 必须保存资格门控、局部证据量、候选分差和候选召回范围。q 仅用于解释与调度；第一版不用它作为计票权重，也不用简单的 q>0.9 做唯一放行条件。

### 3.3 来源追踪保存的不是完整多视图证据

现有 trace 保存的是 3 cm 格子和计数，不是原始像素、精确世界点、实际相机视图或验证帧列表。P1-B 的逐观测复核不能只读该 trace 就声称做了多视角核验。

原始 mask、深度、位姿和相机模型仍是核验来源；按已固定采样策略重建或缓存 `sample_pixel_indices/world_xyz`，并保存采样配置及输入哈希。

### 3.4 P0 发布器拒绝未定身份

`rebuild_p0_from_regions.py` 要求关联表精确覆盖检查点，且每个局部区域都能查到正整数实例 ID。`frame_instance_keys()` 同样拒绝未映射 ID。

不能只将 pending 写成 -1 后直接沿用现有入口；也不能把 pending 行删掉来绕过一致性检查。必须增加状态感知发布路径，同时保留全量来源核验。

### 3.5 版本不能用“当前版本减一”推断

`save_candidate_support_trace()` 当前写 `map_version_queried=self.map_version-1`。这在一次处理帧只有一次版本推进时可成立；P1-B 有延迟放行、审核与后续修订后，不再是安全约定。

改为评分时显式捕获并传递 `read_map_version`。禁止在导出时反推它。

### 3.6 修订日志缺少延迟提交所需的双时间

当前 REASSIGN 记录的 `frame_id` 是原观测帧，不足以说明“什么时候作出修改”。P1-B 必须区分观测发生时间与身份解释生效时间。

## 4. 设计边界：解决什么，不解决什么

P1-B 解决：已经识别为模糊、局部证据过少或候选不完整的观测，不立即污染正式身份；后续有证据时能恢复到正确旧 ID 或建立可追溯新实例。

P1-B 不承诺：

- 纠正所有高置信但系统性错误的关联；
- 从从未形成独立观测的小物体中凭空创造真实实例；
- 把一个混合 mask 自动切成多个物体；
- 自动合并/拆分所有已发布实例；
- 解决动态物体运动、位姿优化或 TSDF 动态重融；
- 把非校准的支持比转换为客观真值概率。

第一版不新增 VLM/CLIP/新分割器、全局 EM、多假设树穷举、全局匈牙利一对一、时间衰减或新空间去噪。保持原前端、采样、3 cm 检索、概率几何公式和 v3。验证原始视图属于暂定解决机制，不是更换前端。

## 5. 三套状态，不能混称 CONFIRMED

### 5.1 Observation 状态

```text
UNPROJECTABLE    无合法投影；来源保留，不给正式 ID，不提供身份票
PENDING_BIND     有旧候选，但归属未决
PENDING_BIRTH    暂无充分旧实例解释；仅是新实例假设
ACCEPTED        已获单一操作性 persistent ID，可进入正式证据
```

`PENDING_BIND` 和 `PENDING_BIRTH` 之间可以随新增证据变化，必须记录事件，不加新的观察票。

### 5.2 Pending packet 生命周期

```text
ACTIVE -> DORMANT -> REACTIVATED
ACTIVE -> PARTIALLY_RESOLVED -> RESOLVED
```

`packet_id` 是独立命名空间，例如 `Q00042`，不是正式实例 ID。packet 是审核工作集，不是已证明同物体的 cluster。可以保存可撤销的观测相容边；不能通过图的连通性无条件把所有成员绑定为同一物体。

同一物体后续出现时，应优先找已有 packet，避免每帧新建一串临时 ID。多个 packet 都可能匹配时保留不确定关系，不进行不可逆并集。若同帧碎片可进入同一工作集，它们仍只能贡献一个“不同帧”确认计数。

### 5.3 Surface 状态

继续保存原 P0 四状态，并额外保存身份未决覆盖与发布原因。它与上面的 ACCEPTED、PENDING 是不同层级。

```text
原 P0: U / T / CONFIRMED / CONFLICT
额外: pending_identity_coverage、unresolved_count_upper、publication_reason
```

有原始 mask 支持但身份未决，不等同于没有观测。

## 6. 数据结构与权威来源

### 6.1 RawSupportRecord，不可变

```python
@dataclass(frozen=True)
class RawSupportRecord:
    observation_id: str
    source_frame_id: int
    mask_local_id: int
    source_mask_sha256: str
    depth_pose_intrinsics_hash: str
    voxels: tuple[tuple[int, int, int], ...]
    sampling_config_hash: str
    support_hash: str
    # 精确重投影信息可惰性重建/缓存，但必须可按上述来源还原。
```

不以伪造空 voxels 的办法阻止 pending 计票，那会丢掉真实证据。

### 6.2 AssignmentRecord，可版本更新

```text
observation_id
status
persistent_instance_id: int | null
packet_id: str | null
assignment_version
first_decision_frame_id
last_decision_frame_id
max_evidence_frame_id
read_map_version
active_since_map_version: int | null
candidate_records
reason
origin_packet_id: str | null
```

### 6.3 DecisionEvent，追加且不可覆盖

```text
event_id / transaction_id
type: DEFER / LINK_PACKET / ACCEPT_BIND / CONFIRM_BIRTH / ACTIVATE /
      MARK_DORMANT / REACTIVATE / REASSIGN / ABORT
observation_id / selected_observation_ids
old_status / new_status
old_id / new_id
source_frame_id
decision_frame_id
max_evidence_frame_id
read_map_version / written_map_version
old_assignment_version / new_assignment_version
witness_observation_ids / witness_frame_ids / evidence_group_ids
excluded_origin_packets
rule_version / parameters_hash / reason
```

逐帧原始 `decisions_at_arrival.jsonl` 保留原判断；另导出 `assignments_asof_<t>.jsonl` 和最终当前表。不把早期模糊分数伪装成后续放行的分数。

### 6.4 两个可独立重建的空间索引

- 正式索引：只由 ACCEPTED 观测派生，沿用帧级计数与引用数。
- 暂定索引：从 pending 原始支持重建，只返回待查 packet/observation，不进入正式 P(v,k) 的分母或分子。

持久 ID 只在确认新生时分配，沿用 P1-A 高水位；packet ID 使用独立高水位或稳定 UUID，不能用正式 ID 冒充归属。

## 7. 当前观测的分流规则

### 7.1 高置信旧实例快速路径

继承 P1-A 资格门控，额外检查：

1. 第一候选与可信竞争者有足够分差；
2. 不是完全依赖缺失可见性时的几何回退；
3. 当前接触区域有足够的重复帧支持，不能用整个实例在远处的长历史替代；
4. 未评候选不足以改变当前结论，或者已完成候选扩展；
5. 没有明确的跨视图矛盾。

可复用现有 `source_single_frame_fraction` 建局部成熟度：

```text
mature_fraction(o,k) = 查询命中格子中，最大值来源对 k 有至少2帧支持的比例
```

它不是当前 mask 所有点都必须稳定；新表面仍可随可靠观测扩展。不要要求每个新点都有两帧旧支持，否则实例无法长大。

成熟度的默认开发起点可以是 >=0.5，分差先沿用0.05；这些是待验证参数，不是已确定最优值。第一版先记录分布，观察小物体的拒绝比例与后续回收比例。

### 7.2 风险路径

以下情况进入暂定：

- 低分差；
- 旧候选存在但绝对匹配弱；
- 一帧纯证据主导；
- 缺可见性核验；
- 候选截断可能影响决定；
- 不同视角对同一区域支持不同身份。

无有效投影进入 UNPROJECTABLE，不能当作可靠新生实例。

已有实例同帧接收多个 mask 的能力继续保留。多对一只作为风险信号之一，不能一律拒绝或改为一对一，否则会重新制造过拆分。

## 8. 候选完整性：避免“前三名里看起来很确定”

第一阶段仍先执行 A2 的空间 top8 快速评分；但 P1-B scorer 应提供完整 `CandidateRecord`，不只返回前三名日志。

空间候选的原始覆盖率 G_binary 给出以下综合分上界：

```text
U(k) = 0.5 * G_binary(k) + 0.5
```

因为 G_prob<=G_binary、V<=1，而无可见性时 S=G_prob<=U，所以该上界覆盖现有两种评分分支。

如果未评候选的 U(k) 仍可能达到第一名减分差阈值，需要进一步评分。无法在预算内排除竞争者时，标记候选不完整并暂定，不将“没算到”解释成“没有”。

实现应使用未舍入覆盖率计算上界，并保留例如2e-6的数值容差；不要为此改变A2兼容路径原有的舍入行为。原始几何覆盖率明显低于继承资格门槛的候选可以排除。候选扩展须记录 `candidate_count_retrieved/scored/truncated`，并增加禁用扩展的消融，避免把扩展收益都归给延迟提交。

这只完善当前局部候选范围，不保证找回完全无空间交集的真实实例。

## 9. 暂定观测如何获得新的身份证据

### 9.1 暂定包的组织不产生正式身份结论

新观测优先查询已有 packet 的原始空间支持。用双向、可见部分上的原始 mask 回投影相容度建立临时边，而非单向“距离一个大实例很近”。

对观测 o 与另一帧观测 r：

1. 从 o 的原始像素与深度恢复真实世界点；不是3 cm格子中心。
2. 投到 r 的相机。
3. 只统计在图内、深度一致、未被遮挡的支持。
4. 计算其中落入 r.mask 的比例 A(o->r)。
5. 反向计算 A(r->o)。
6. 双向都有足够有效支持且相容，才记录正相容边；无可见交集记 UNKNOWN，不记反对。

可以用 min(A(o->r), A(r->o)) 作为保守相容分数。分母只包含可见部分，不能用整个旧大物体体积惩罚当前局部观测。

对一张大 mask 在另一视图覆盖多个独立小 mask 的情形，应记录 MULTI_REGION_CONTRADICTION。不得只挑其中一个小 mask 作为支持，然后把整个大 mask 绑定过去。这类例子继续暂定并输出将来的区域修复 ticket。

### 9.2 绑定旧实例：身份锚点 + 原始视图相容

一个有效身份锚点 r 必须：

- 来自另一真实处理帧；
- 本身能在不使用被审 packet 支持的正式状态上，可靠匹配旧 ID k；
- 与待决观测 o 有原始视图一致性；
- 不是同一图像或同一近重复视角被反复计作多个新证据；
- 没有更强的反对证据。

开发起点：待决观测获得至少两个有新增信息的锚点帧支持同一个旧 ID，再放行。第一版不把 softmax 分数跨帧相乘、不把重复重判累加成证据。

`evidence_group_id` 用于合并近重复视角；相机观察方向改变或新增可见支持可作为新增信息判据。它减少明显重复，不声称不同组在统计上严格独立。小物体核验需要比例与最小可见支撑结合，不机械照搬大物体需要大量像素的门槛。

有历史独立锚点也可使用，只要其时间不晚于当前决定时刻，且不依赖被审 packet。未来观测只有实际到达后才可用。

### 9.3 防自证是必要条件

审核 Q 时，证据视图排除 Q 中所有观测，以及已经由 Q 放行、会直接形成自支持的成员贡献。

```text
EvidenceView(exclude_origin_packets={Q})
```

排除必须按引用数重建局部有效计票，不能粗暴删除整个体素或整个实例，也不能把同帧其他独立观测仍提供的票一起删掉。

同一packet内另一帧的原始图像可以作为几何见证；但它的临时ID不是身份依据。它若要为旧ID背书，也必须在去除Q贡献的证据视图上独立获得可靠关联。当前 packet 尚未放行时，隔离自然防止直接自证；部分放行后，显式来源排除尤其重要。此规则不能排除所有间接、跨多轮传播的依赖或早期系统性错误；完整因果依赖修复属于后续层。

### 9.4 没有旧实例时怎么启动

不能要求新物体先有已确认旧实例支持，否则空地图永远无法启动。

PENDING_BIRTH 通过独立的“新生确认”路径：

- 至少两个实际处理帧具有一致原始区域支持；
- 有新增视角/可见支持，而不是重复播放同一观测；
- 复查后没有足够可靠的旧 ID 可以解释；
- 未出现明显混合 mask 或跨物体矛盾。

满足时分配一个正式 persistent ID，原临时 packet 到正式 ID 的映射写入事件。两个视图是贴近现有P0最少两票的开发起点，不是可靠性证明；必须报告只被观察一两次的小物体受影响程度。

“找不到旧候选”不保证新物体，仍可能是视角变化或几何漂移造成的 false split。此边界不可掩盖。

### 9.5 逐观测放行，禁止整包盲目搬迁

Q里共有10张mask，其中7张一致支持k，另外3张不一致：

```text
只放行那7张通过逐观测核验的mask。
其余继续pending/拆分审核工作集。
不将10张统一迁移，也不拆改原mask。
```

同理，Q中出现一张高分观测，不能据此自动确认所有旧成员。审核组的存在不等于同物体事实。

## 10. 逐帧执行的确定顺序

```python
def process_frame(frame, observations):
    validate_inputs_and_duplicate_ids()
    snapshot = freeze_committed_state()  # 显式版本，不是运行结束后减1

    raw_records = project_without_changing_source_masks(observations)
    candidates = score_all_current_against(snapshot)
    arrival_actions = gate_to_accepted_or_pending(candidates)

    # 组包只使用旧pending索引及本帧不可变观测；不能让本帧新身份进入本帧评分。
    packet_updates = propose_packet_links(snapshot.pending, raw_records)
    review_set = schedule_changed_packets(packet_updates, snapshot)
    review_actions = validate_using_raw_witnesses(
        review_set, snapshot, current_raw_observations=raw_records,
        exclude_self_support=True,
    )

    # 本帧可提供新证据；它的身份锚定必须基于同一个旧snapshot。
    # review_actions彼此不能成为同一事务中的自洽循环证明。
    transaction = prepare_validated_transaction(
        arrival_actions, packet_updates, review_actions,
        expected_map_version=snapshot.version,
    )
    atomically_apply(transaction)
    publish_versioned_status_and_incremental_deltas()
```

允许在当前 t 时刻使用刚到达的真实观测来审核旧 pending；不允许使用 t 以后数据，不允许先试加票再用它证明试加票正确。

同一事件可以更新多条旧观测，整个批次必须预检查唯一性、状态、版本、目标ID与证据时间，再变更。失败时不留半个packet已放行、半个缓存未同步的状态。

## 11. 放行计票与双时间

例如观测源帧100和105，到110才确定归属ID7：

- 体素与P0原始票的去重键仍用100、105；
- 生效时间记110；
- 当时输出的100/105前缀不能被追溯改写成已经知道ID7；
- 查询110时可以利用100、105的真实历史支持；
- 再次复核同一观测不增加票。

必须满足：

```text
source_frame_id <= max_evidence_frame_id <= decision_frame_id
snapshot_asof(t) 只包含 decision_frame_id <= t 的生效事件
```

`map_version` 是状态序列，不代替物理帧/处理帧时间。检查点要保存处理步号、原始帧号、事件高水位、正式ID高水位、packet高水位、active/pending状态与调度去重指纹。

## 12. 状态感知表面发布：不能通过藏起未决证据制造高置信

P1-B 的身份隔离不等于把原始未决证据抹掉。

反例：表面点已有3票A，另有2条未决观测覆盖此点。只对accepted票做P0会显示A=100%，但未决观测未来可能支持别的实例。暂定区域也不能被伪装成U，再用扩散直接吸收。

### 12.1 保留两个输出用途

- `p0_accepted_only`：严格沿用原P0，只使用已激活票；作为兼容诊断，不单独用来证明不确定性已消失。
- `instance_surface`：最终主输出，使用未决支持保护的发布规则；这才进入P1-B主评估。

### 12.2 一个可审计的保守发布规则

令a_k是原P0按 `(frame, point, accepted_id)` 去重后的票；A=sum_k a_k。令u是当前覆盖该点的、**不同未决原始observation**的数量上界，同一observation对此点只算一次。

u不按暂定重判次数增加；同帧若有多张不同未决mask，它们将来可能属于不同实例，所以不能未经证明全部压成一票并声称它是最坏情况界。

定义：

```text
worst_case_vote_share(k) = a_k / (A + u)
```

这不是概率后验，也不是给unknown添加实例票；只是固定当前accepted赋值、不改原mask的条件下，将未决支持全部视为潜在竞争者时的保守发布比例。

主输出发布top1仅当：

```text
top1 accepted票 >= 2
worst_case_vote_share(top1) >= 0.67
```

若u=0，退化为原P0发布规则；若A=0且u>0，明确为PENDING_IDENTITY；若A=u=0，才是确实无这类实例观测支持。这个上界会保守，特别是同帧多碎片区域，因此要单独测覆盖损失，不能隐藏代价。

统计正式identity概率时仍只使用accepted票；u不进入3cm身份证据的分母，避免unknown质量改变当前关联公式。

### 12.3 扩散边界

首轮主比较全部使用原生输出，不同时调扩散。需要后处理比较时，必须保护因pending而暂缓发布的区域；不得让扩散绕过身份审核。

若现有扩散入口增加了这类保护，应明确称为“相同扩散参数＋pending语义保护适配”，而不是声称完全未经修改的同一流程。补全身份不得回灌为原始观测票。

### 12.4 在线几何边界

当前最终TSDF区域缓存可继续用于完整序列结束后的固定几何重发布。真实在线前缀必须用当时几何或严格因果的前缀投影；最终TSDF不能帮助早期关联、锚点验证或前缀质量主张。

## 13. 有界调度，不把P1-B变成全历史重优化

每个处理帧仍执行当前观测关联；只对以下packet重查：

- 新观测命中其空间范围；
- 候选旧ID相关局部支持更新；
- 出现新的有信息视角或矛盾；
- 明确的审计重查请求。

保存 `evidence_fingerprint`，包括新增见证ID、相关支持版本、候选集合、规则配置。证据没变化，重查不能增加确认计数。

开发起点：活动窗口20个处理帧，每帧最多复核32个packet，按风险、等待时间和局部覆盖公平调度；不能按体积/像素数让小物体一直排不到。预算用可重复的任务数量，避免wall-clock超时令相同输入产生不可重复决策。

超出活动窗口进入DORMANT，原始支持仍在磁盘账本中；后续重观测可激活。不因超时强行建新ID、不删除证据、不在序列结束时无条件flush为确定归属。

这些窗口/配额只是可运行起点，正式值需依照待决数量和延迟分布确定。

## 14. 推荐模块划分

保持A2行为不变，新增独立 `DeferredAssociator` 路径。不要用现有 `--improved-association` 偷偷开启P1-B：那个开关控制的是旧候选回退和采样默认，不是延迟身份机制。

| 文件/模块 | 职责 |
|---|---|
| 新 `deferred_association.py` | P1-B整帧编排、快速门控、packet状态机 |
| 新 `assignment_ledger.py` | 全量RawSupport、归属状态、事件与双时间 |
| 新 `pending_observations.py` | 独立pending索引、临时相容边、惰性唤醒 |
| 新 `association_review.py` | 原始视图见证、逐观测审核、来源排除、证据去重 |
| `identity_evidence.py` 小扩展 | 正式贡献activate/deactivate事务接口；保留A2计票语义 |
| `association.py` 小重构 | 可复用的只读评分结果、显式read_version、完整候选诊断 |
| 新 `run_p1b_association.py` | 独立入口；A2与B0兼容运行 |
| 新/扩展 `rebuild_p1b_surface.py` | 状态感知发布、pending覆盖与保守发布界 |
| 新 `run_p1b_validation.py` | A2/B0/B1/B2、因果前缀、注入错误和身份/覆盖指标 |

建议先保持 `FrameIdentityEvidence.entries` 为正式贡献的派生账本，不直接承担全量pending来源。新增公开的activate/deactivate或事务接口；不要让上层到处调用 `_remove_votes()` 私有函数。

生产入口的snapshot格式升级为独立P1-B版本；加载A2检查点时所有既有观测标记ACCEPTED、原事件与ID高水位保留。新建正式ID与延迟放行都需要能保存/恢复；恢复不能因packet丢失而每帧重建临时ID。

## 15. 实施顺序与消融

### B0：工程兼容层

新增全量来源表、归属状态、两个索引、双时间和状态感知导出，全部决定按A2直接接受；不启用暂定。

验收：相同输入下A2/B0的原有决定、计数、原生P0数组逐项一致；新元数据允许增加。u=0的主发布必须等同原P0。

### B1：隔离对照

开启风险门控、临时包连续组织、独立新生确认；对旧ID的模糊绑定不启用后续解歧放行。

B1是“只隔离不回收”的诊断消融，不是推荐最终方法。新生确认必须保留，否则空图没有正式实例、比较会变成全拒绝。

### B2：完整P1-B

在相同B1规则和出生机制上，开启新增见证、来源排除、逐观测延迟放行。B2才是目标实现。

额外小规模对照：

- 只延迟新生、不延迟旧ID模糊匹配的track-confirmation对照；
- 禁用自证排除，用于说明为何“等几帧”不够；
- 固定原top8与候选安全扩展，区分候选召回收益；
- accepted-only发布与未决保护发布，公开覆盖/风险取舍。

不要在全部八场景上逐场选最优阈值。先固定开发场景与参数搜索范围，再冻结设置评估其他场景。已看过八场景结果，不能称其为从未触碰的严格holdout。论文泛化主张需要另外数据验证，但不因此阻塞本轮基础实现。

## 16. 必须通过的行为测试

1. B0关闭延迟时A2等价。
2. pending登记不改变正式C、索引、P0身份票。
3. 同一packet连续出现不会每帧新建临时对象。
4. 重复处理同一帧/同一观测拒绝且无部分变更。
5. 同帧多个碎片不增加不同帧或不同见证组计数。
6. 一帧纯支持、q接近1，不能伪装成成熟证据。
7. 几何回退缺视角验证能进入pending。
8. 原第四/第九候选可能改变结论时，扩展或暂定，而非伪确定。
9. 空图冷启动经过新生验证能生成正式实例。
10. 一帧瞬时mask保留为未决，不被当成长期物体。
11. 新证据支持旧ID时逐观测放行，原来源帧计票。
12. 反复审核同证据不增加票和确认次数。
13. Q成员不能仅凭临时身份或既往放行结论为Q自证；重新核验其原始视图与外部独立身份依据需明确区分。
14. 删除Q的贡献时不误删同帧其他独立观测仍支持的票。
15. 一包7张正确、3张混合，不能10张一起绑定。
16. 两个相邻同类对象不因近邻或语义相同自动merge。
17. 同一物体同帧多mask仍可经核验归入同一正式ID。
18. 无共同可见区域记UNKNOWN，而非反对。
19. 支持3票A加2条未决时，不能仍称无竞争的100%稳定。
20. pending-only点区别于无观测U，扩散不能绕过保护。
21. 延迟放行在旧前缀不可见，在决定时刻可见。
22. 加入未来输入不改变已保存过去前缀。
23. 检查点恢复包含pending、见证、packet、事件及所有高水位。
24. 增量状态与全量来源重建一致。
25. 事务中途故障回滚，不留半写状态。
26. 过期packet不强行确认、不删除证据，重观测可唤醒。
27. 持续错误、没有独立假设时保留失败，不伪造自愈。
28. 已ACCEPTED观测不因为argmax变化自动重派；P1-B不冒充历史修复。

## 17. 实验成功判据

原生v3固定口径继续测AP50、F1、PQ、merge/split/duplicate、小物体召回、地图覆盖。正式/暂定实例数分别报，不用累计ID数替代最终物体数。

新增P1-B诊断：

- 暂定比例、累计暂定量、最终残留量；
- 延迟放行正确率与误放行率，标明是否人工/GT离线定义；
- 仅放行样本的正确率与全部样本覆盖率配对报告；
- 延迟中位数/P95，源帧到决定帧按处理步和原始帧分别计；
- 小物体暂定比例、放行比例和未解决FN；
- 一次注入错误后的后续污染观测数、错误支持增长、是否恢复；
- 时间、内存、审核数量与积压。

A2与B2必须从空图独立运行。不能先用完整A2最终结果作为B2早期可靠地图。

受控试验至少含：

1. 已有独立正确ID，一次诱导误关联；
2. 某区域处于两个旧ID竞争中，后续视图解歧；
3. 初次出现的新小物体，检查冷启动与召回；
4. 高置信持续错误且无正确独立假设，保留不能解决的边界；
5. 原始mask真实混合，证明仅整mask归属无法完成所需修复。

同一预算、同一发布协议下，B2必须优于只隔离的B1；若只提高已发布部分精度、整体覆盖和小物体召回显著下降，不能认定方案成功。

## 18. 与CVPR论文主线的关系

概率体素和tentative track本身已有先例。OpenVox采用实例关联与体素更新分离，最终仍进行最高分关联；ThinkGraphs采用概率体素加暂定轨迹确认，并有merge-only Critic；Voxeland已有概率不确定性与几何拆分/合并。

因此本轮不能将“概率＋等两帧＋split”写成独有贡献。P1-B是可修订体系的预提交部分：先保留未决来源，再依据可追溯新证据提交。后续历史层修改已经提交的错误，撤销真实支持并重放后继依赖。

最有价值的共同接口是：

```text
提出解释 -> 独立来源核验 -> 逐观测约束 -> 版本化事务 -> 局部统计重建
```

P1-B与后续P2共用这条接口，但P1-B不自动改已ACCEPTED身份。不要为了把基础关联调到最好而无限延长本阶段。

## 19. 可直接交给Codex的任务约束

从 `55c099c38a34752c0daa5d7f3a430f120f4001b1` 新建P1-B开发分支；不要改main，不覆盖A2结果。先实现B0并证明等价，再实现B1和B2。按本文件的状态、隔离、见证、双时间和发布合同落地。

特别禁止：pending仍调用正式_commit；临时ID加入正式概率场；q作为真值概率或反复计票；整包一键绑定；用自己刚写的支持确认自己；超时强制建新ID；最终TSDF参与早期决定；删除未决来源以通过P0检查；pending灰区被扩散绕过审核；改变v3版本掩盖分数变化。

交付代码、测试、B0等价报告、逐帧日志、状态/候选/见证来源、检查点恢复报告、v3原生结果和受控传播审计。参数来自开发起点和固定开发集选择，不声称已有最优阈值。

## 20. 参考来源

### 审阅的固定代码快照

- 验收报告：[p1a_validation_20261001.md](https://github.com/liuwenjing613-maker/3D-mapping/blob/55c099c38a34752c0daa5d7f3a430f120f4001b1/revisable_instance_map/docs/p1a_validation_20261001.md)
- 计票核心：[identity_evidence.py](https://github.com/liuwenjing613-maker/3D-mapping/blob/55c099c38a34752c0daa5d7f3a430f120f4001b1/revisable_instance_map/src/revisable_instance_map/identity_evidence.py)
- 关联与版本：[association.py](https://github.com/liuwenjing613-maker/3D-mapping/blob/55c099c38a34752c0daa5d7f3a430f120f4001b1/revisable_instance_map/src/revisable_instance_map/association.py)
- 发布接口：[rebuild_p0_from_regions.py](https://github.com/liuwenjing613-maker/3D-mapping/blob/55c099c38a34752c0daa5d7f3a430f120f4001b1/revisable_instance_map/tools/rebuild_p0_from_regions.py)
- 原测试：[test_identity_evidence.py](https://github.com/liuwenjing613-maker/3D-mapping/blob/55c099c38a34752c0daa5d7f3a430f120f4001b1/revisable_instance_map/tests/test_identity_evidence.py)

### 原始研究资料

- OpenVox, [arXiv:2502.16528v1](https://arxiv.org/html/2502.16528v1)，关联与地图演化分离、计数式实例分布。
- Think While You Map, [arXiv:2606.31471v1](https://arxiv.org/html/2606.31471v1)，概率体素关联、暂定轨迹确认、merge-only Critic。
- Voxeland, [arXiv:2411.08727v2](https://arxiv.org/html/2411.08727v2)，证据不确定性、实例空间拆分与周期合并。

文中隔离工作集、来源排除、未决发布下界与具体接口是针对本仓库的设计建议，不把它们误标成上述论文原有算法。
