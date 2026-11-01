# 整场景结束后统一执行 B

当前发布策略为 `scene-final-B-20261101`，配置见 [`scene_completion/policy.json`](../scene_completion/policy.json)。适用于 `P1-A1-strict-vote`（no_repair）、main、auto-v3-swap、online-v1 和 online-v2 的后续整场景实验。所有流程都是：**完成整场景建图 → 完成该流程全部最终处理/末次修复 → 执行一次 B → 使用 B 产物做最终评价和实例 PLY 导出**。C 不启用。

B 不放入建图循环、50/100/200/300 帧检查点、ROI/种子/SAM/E1 处理或修复提交中。在线 400 帧的表面重建本身也不足以触发 B，必须等最后一次修复提交结束。B 输出单独保存，不写回关联状态、历史 Mask、票数或在线检查点，不参与后续续建。完整场景目前明确定义为 Replica 原始帧 `0,5,…,1995` 共400帧；其他采样范围需另建明确的协议，不允许将短前缀改名为 final 来通过检查。

## 已验证的 B 数值策略

`scene_completion/core.py` 与 2026-11-01 八场景实验的 B 源码逐字节相同。只处理最终地图 `instance_id == -1` 的点，种子只取该次最终证据中的原始 CONFIRMED 点；使用20邻居、2.5cm半径、法向余弦≥0.92、双向切平面偏差≤3mm、同ID占比≥90%、至少6个同ID邻居。T 还须与原始 Top-1 一致；CONFLICT 保持未分配。只执行一轮，不以新补全点作种子，已分配ID不改动。灰点仍可能保留，这不是强制全覆盖。

## 整场景入口（新实验默认）

以下 `whole_scene_driver.py` 代表使用独立代码/结果根适配后的**完整场景驱动**，不是单个检查点脚本。旧实验路径固定快照和结果锁仍是历史记录，不能直接重跑覆盖。新入口包裹完整驱动，子进程成功返回后才执行 B；任一步失败都不发布 B。

对于 main、auto-v3-swap、online-v1、online-v2，最后处理阶段须完成与现有 `commit_complete.json` 相同的可审计提交格式：

```bash
python -m scene_completion committed \
  --workflow online-v2 --scene room0 \
  --root /data/chenkejun/CVPR/results/NEW/room0/400/A \
  --manifest /data/chenkejun/CVPR/results/NEW/room0/scene_complete.json \
  --out /data/chenkejun/CVPR/results/NEW/room0/final_B \
  -- python /home/chenkejun/CVPR/experiments/NEW/whole_scene_driver.py
```

入口检查完整 mapping 帧列表、末次400帧修复提交、final map SHA256、证据审计及逐点几何一致性。已有完整提交可省略末尾 producer 命令，只执行最终发布。非此提交格式的完整驱动使用下文通用接口；禁止凭一个“PASS”字符串伪造提交。

无修复完整驱动执行完严格投票和已有 final 后处理后，再接 B：

```bash
python -m scene_completion no-repair --scene room0 \
  --mapping-receipt /data/chenkejun/CVPR/results/NEW/room0/mapping_complete.json \
  --surface /data/chenkejun/CVPR/results/NEW/room0/final/instance_surface.npz \
  --native /data/chenkejun/CVPR/results/NEW/room0/native/instance_surface.npz \
  --evidence /data/chenkejun/CVPR/results/NEW/room0/surface_evidence.npz \
  --normal-geometry /data/chenkejun/CVPR/results/NEW/room0/geometry/surface.ply \
  --manifest /data/chenkejun/CVPR/results/NEW/room0/scene_complete.json \
  --out /data/chenkejun/CVPR/results/NEW/room0/final_B \
  -- python /home/chenkejun/CVPR/experiments/NEW/whole_scene_driver.py
```

mapping 凭据必须含 `status: PASS` 和实际完成的 `frame_ids`。该命令等待整个驱动，包括最终严格投票/原 final 导出完成后才消费凭据。不要传只建图、尚未执行 final 后处理的命令。

通用驱动可在**最外层整场景流程末尾**调用 `scene_completion.pipeline.write_scene_complete(...)`，提供实际处理的帧列表、四个输入文件及最终处理完成标志；有修复时还须明确 `final_repair_complete=True`。随后通过 `python -m scene_completion run --manifest … --out … -- python whole_scene_driver.py` 自动收尾，或在驱动末尾调用 `finalize(manifest, out)`。该 hook 是生产者的完成契约，不能放在检查点循环中，也不能把计划帧数冒充实际帧数。

`write_scene_complete` 的输入名为 `surface/native/evidence/normal_geometry`；文件必须绝对路径，manifest 写入时绑定SHA256。`normal_geometry` 支持保留点序及法向的PLY，或含 `xyz_m/normals` 的NPZ。不读取GT，不重新分割，也不重新运行修复。

## 产物、续跑及评价

`final_B/instance_surface.npz` 是新的最终展示/评分输入，保留原坐标、RGB和所有原字段，并增加 `scene_completion_policy` 标记；`scene_finalization.json` 的 `final_surface` 是下游唯一发布指针。原 commit 的 `final_map` 保持 pre-B，继续用于状态审计和历史续建，不能拿它冒充 B 结果。`completion_stats.json`、`local_diagnostics.npz` 保存增量和拒绝原因。

仅在 `scene_finalization.json` 为PASS时认为发布完成。相同 manifest、源码和输入重复调用 `finalize` 不再计算；源文件/输出损坏、其他输入复用同一目录、再次以 B 产物作为输入均拒绝。并发有独占锁；失败保留现场，不自动覆盖，请检查后使用新目录重试。重复运行整个 producer 用新实验目录，已有 producer 结果续收尾用 `finalize` 或不带 producer 的 `committed`。

所有后续方案统一采用这份 B 策略；原始 pre-B 与历史指标保留明确标记。评价仍从服务器 main 的当前默认 `unified_eval` R2 入口运行，350GT、`frozen=false`，成对比较固定相同几何/GT/索引。此次发布没有重新测量新的种子边界规则下的指标，既有 B 实验也不能冒充该规则的新结果。

安装核心依赖 `numpy/scipy`；读取PLY另需 `open3d`（仓库的 `completion` 可选依赖）。验证命令：`python -m unittest discover -s tests -p 'test_scene_*b.py'`。数值回归和场景结束边界测试的记录见 `experiments/scene_final_b_20261101/validation.json`。
