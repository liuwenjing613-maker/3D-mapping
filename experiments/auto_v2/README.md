# auto-v2：已采用的种子筛选链

基于 `auto-v1` 的 `35bff6768579c7ad2cd4d4b42246ae9a6a291384`。保留原有候选生成与 Mask；在最终 90 个主种子之后依次应用下表五项筛选。纯自动规则得到 33 个；2026-10-19 用户明确恢复本轮的 G017、G025，实际送原版 E1 的是 **35 个**。这两个恢复只属于 room2 本轮人工配置，不是通用自动规则。

| 顺序 | 本轮采用的规则 | 输入 → 保留 |
|---|---|---:|
| 1 | SegFormer 整图预测；原 Mask 内高置信墙／地／顶／地毯像素占比 ≥90%。像素最大概率 ≥0.7。达到条件排除，其余保留。 | 90 → 72 |
| 2 | 共同可见 IoU ≥0.8；以完整支持点数较大的固定主种子吸收直接匹配种子，不传递合并。不使用单独可见比例或20点门槛。 | 72 → 56 |
| 3 | B：只看种子来源帧，p、r 同时 ≥95% 跳过修复。冲突数量不参与判定，不加最小支持点门槛。 | 56 → 45 |
| 4 | 完整 TSDF 支持点被更大种子包含的比例严格 >70%，排除小种子。不裁到共同可见区域。 | 45 → 42 |
| 5 | p 严格 >10% **且** r 严格 <20% 时排除。边界相等保留。 | 42 → 33 |
| 本轮人工配置 | 恢复 G017、G025；其余 G069、G071、G074、G075、G077、G082、G086 仍排除。 | 33 → 35 |

p = C/N，r = C/I。N 为原种子来源帧可见的完整 TSDF 支持点数，包含未分配点；I 为与种子重叠最多的正实例在该帧的全部可见点数；C 为交集。I 不能裁进种子内部。两阶段复用同一严格一票、未修复地图：`p1a1_one_vote_20261009/ablation/room2/abstain/final/instance_surface.npz`，SHA-256 为 `3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3`。

## 使用入口

在仓库根目录运行：

```bash
python -B -m unittest discover -s experiments/auto_v2 -p 'test_*.py' -v
python -B experiments/auto_v2/run.py --out /path/to/new_output
# 需要单独检查纯自动33个版本时：
python -B experiments/auto_v2/run.py --automatic-only --out /path/to/new_automatic_output
```

入口重放原有实测计数，逐项重新计算五阶段决定，不加载历史决定作为算法输入。`reference_decisions.json` 只在计算结束后验证与历史结果逐项一致。输出包括逐阶段判断 `selection.json`、纯自动清单 `automatic_seed_manifest.json`、本轮最终清单 `seed_manifest.json` 和验证回执。拒绝覆盖已有结果目录；不执行模型、追踪、E1 或评价。

`seed_manifest.json` 保留原始完整 group、Mask 路径和稀疏 `original_track_id`，不重编号、合并或裁剪 Mask。本目录只提交小型计数和来源记录，运行清单需通过 `mask_source_directory` 或原始 r4 目录读取 Mask。不要把稀疏原编号重新编号后接入旧 E1。

服务器上的完整来源核验：

```bash
python -B experiments/auto_v2/run.py \
  --out /data/chenkejun/CVPR/results/auto_v2_new_run \
  --original-manifest /data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4/seed_manifest.json \
  --mask-root /data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4 \
  --verify-server-sources
```

`--mask-root` 校验全部 90 张原始 Mask 文件哈希，`--verify-server-sources` 另外核验每份服务器原始结果、固定地图和证据文件。只在本地使用提交的计数重放时，回执会明确 `source_binary_masks_verified=false`，不会误报已经访问服务器原始 Mask。

## 文件与来源

- `pipeline.py`：五阶段规则和明确的本轮人工覆盖；整数比较阈值，不根据显示百分比作判断。
- `config.json`：当前唯一采用的规则。更改为其他规则会拒绝复用这份测量，不会悄悄忽略配置。
- `fixtures/room2_20261019/measurements.json`：90 张种子的语义计数、72 张保留种子的全部 2,556 对测量、56 张种子的 B 计数；不含人工标签和预设判定。
- `manual_overrides.json`：只对已确认的 42 个输入和原来的 9 个 p/r 排除项生效，校验 G017/G025 的原始身份和 Mask 哈希。
- `source_lock.json`：原数据及提交计数的哈希。`prepare_inputs.py` 可从服务器冻结结果重新提取这些小型文件，既不推理也不生成图片。
- `dependencies/strict_vote_annotation_20261010/common.py`：去重测量实际使用的可见投影代码，逐字节核验原始来源哈希后归档。
- `evidence/room2_20261019`：实际服务器重放的35个清单和回执，以及当前 E1 运行清单的只读身份比对。自动阶段分别与已有结果完全一致。

重新提取本轮输入：

```bash
python -B experiments/auto_v2/prepare_inputs.py \
  --artifact-root /data/chenkejun/CVPR/results \
  --original-manifest /data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4/seed_manifest.json \
  --out /data/chenkejun/CVPR/results/auto_v2_new_inputs
```

原始测量与页面实现保持在 `experiments/auto_v1/` 下的 `seed_semantic_filter`、`seed_covisible_dedup`、`seed_skip_repair_b`、`seed_containment_filter`、`seed_pr_filter`。它们是按实际历史运行保存的脚本，包含服务器绝对路径和历史说明。去重目录中的初版可见比例门槛、语义目录里的早期待复核统计只是历史记录；当前采用入口以本目录配置为准。旧 CLIP 对比实验没有串入这条新筛选链。

这次只整理已确认的 room2 流程，未声称完成任意新场景的一键测量。新场景需先准备原始 Mask、固定几何、可见点与整图语义预测，并按这些测量接口生成计数；不能复用 room2 人工恢复项。包含率出现等点数达标或链式排除时会停止并要求确定，当前 room2 没有这些情况。

## 验证和 E1 边界

本地 51 项测试通过：新入口12项，五个原实现分别16、9、7、4、3项。新入口在服务器12项通过，完整来源锁与90张原始 Mask 哈希通过。五阶段的全部种子决定逐项复现，35 个最终种子的 ID、原 track ID、Mask ID、来源帧和 Mask 哈希与实际 E1 输入逐项一致。

实际 E1 清单哈希：`b7188a47abf40974e87f317f3b489d1fe45c4f42fe60bf6b26b9e15196a22ae0`。代码在 `/home/chenkejun/CVPR/experiments/e1_auto35_original_20261019`，结果在 `/data/chenkejun/CVPR/results/e1_auto35_original_20261019`；此分支的清单比对不等于 E1 已完成，不预先报告修复指标。

冻结 E1 的源码和当前 main 评价器均未改动。后续评价从 `/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009` 当前默认入口执行 `object_observed_repair / revision 2`，保持相同几何、GT范围及对应索引。筛选本身不使用 GT。高语义置信度、可见 IoU 或包含率仍可能误判；本分支保存的是本次采用的规则与可复现结果。
