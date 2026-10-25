# online-v2：检查点在线修复与已验证结果

本分支归档当前在线实现与结果，不替换 main 的 `partial＋E1＋swap` 默认基线，不改变冻结数值内核。版本基于 `2e3a39f7`，包含此前 checkpoint-only 优化、完整修复闭环、仅窗户特殊来源、七场景完整 A/B 和 office0 候选分析的已提交实现。

## 执行规则

- 每个建图帧继续读取 RGB-D/原始 Mask，执行持久 ID 关联、更新历史身份映射、融合 TSDF、保存观测。完整表面提取与全部历史严格投票只在检查点执行；未重建帧的完整表面统计不冒充最新地图。
- 400个建图帧为原始帧 `0,5,…,1995`，检查点为50/100/200/300/400。每次暂停建图，完成 ROI→候选/种子→SAM2→Gate→partial＋E1→修复提交，再使用已提交的历史像素、身份与关联状态续建。
- A从所有已到达帧选候选；B只从上一检查点后的新帧选候选。两者均完整重投历史票，不是原版/优化版对照。
- SAM特殊来源仅针对窗户；柜子走普通来源筛选。P/R语义豁免仍保留原有窗户与柜子规则。SAM2、Gate、partial、E1及数值参数不变，不排序重复点、不手工统一候选。
- 精确指纹一致才复用追踪；保留实际推理、缓存、GPU等待及失败重试记录。最多4条分支并发，GPU池0～3；失败停止新增调度。

## 实现入口

| 内容 | 仓库路径 |
|---|---|
| 七场景模板与场景适配生成 | `experiments/online_replica7_window_AB_20261023/build_batch.py` |
| 调度、续跑 | 同目录 `launch_batch.py`、`run_batch.py`、`<scene>/run_branch.py` |
| 空地图起点、建图、提交 | 同目录 `<scene>/initialize_empty.py`、`map_support.py`、`map_continue.py`、`commit_continue.py` |
| 筛选、追踪、修复 | 同目录 `<scene>/selection_run.py`、`track_queue_with_cache.py`、`repair_continue.py` |
| 审计、R2、PLY | 同目录 `audit_branch.py`、`evaluate_branch.py`、`export_checkpoint.py`、`finish_batch.py` |
| 七景完整证据与页面源文件 | 同目录 `evidence/final/`、`FINAL_REPORT.txt`、`viewer.html`、`viewer.js` |
| room0完整在线历史 | `experiments/online_full_scene_room0_AB_20261023/` |
| room0仅400帧新规则重修 | 上述目录 `window_only_repair400/` |
| 四方案对比及原始评分证据 | `experiments/online_v2/analysis/` |
| 本次来源/文件哈希清单 | `experiments/online_v2/release.json` |

这些是已执行实验的路径固定快照，不是一个可在任意工作目录直接冷启动的新 CLI。根目录的 `run_branch.py` 是生成模板，实际执行的是 `<scene>/run_branch.py`。现有结果已完成，不能直接重跑生成器或启动器覆盖它们。重新实验必须先指定新的代码/结果根，并审查路径适配；不能把修改路径后的运行称作本次原始记录。

已执行代码根：`/home/chenkejun/CVPR/worktrees/online-v1-checkpoint-surface-20261023/experiments/online_replica7_window_AB_20261023`。数值解释器：`/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python`。大文件结果根：`/data/chenkejun/CVPR/results/online_replica7_window_AB_20261023`。room0结果根：`/data/chenkejun/CVPR/results/online_full_scene_room0_AB_20261023`。PLY、数据集及权重不重复存入Git；已提交证据包含来源路径与哈希。

评价始终从 `/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009` 的当前默认 R2 入口运行，`object_observed_repair / revision 2`、350GT、`frozen=false`。七景278GT，room0为72GT。只评400帧最终地图，不能用最终GT评在线前缀。

## 结果与实际运行

七景14条分支从空地图开始，70次修复提交全部完成，实际51/101/201/301帧继承核验通过。140份完整PLY已导出，全部文件哈希及70组成对坐标逐元素核验通过；浏览器验证24组切换及退化GT定位。约13小时包含排队、GPU等待、失败恢复与后处理，650组追踪中516组实际推理、134组严格缓存复用，不能宣称是650组冷启动耗时。

适配故障均保留：首批4条分支的catalog路径修正；office3 A的并发状态临时文件竞争修正（文件名增加PID/线程ID，8线程800次写入验证）。这些不改变数值算法。

八场景等权平均（%）；在线room0采用400帧新规则重修，但早期历史不同，故八景仅为当前结果总览：

| 指标 | 无修复严格一票 | main | 在线A | 在线B |
|---|---:|---:|---:|---:|
| AP | 18.77 | 19.91 | 20.86 | 21.13 |
| AP50 | 39.80 | 39.83 | 41.82 | 41.86 |
| F1 | 62.79 | 62.48 | 64.10 | 64.33 |
| PQ | 49.08 | 49.69 | 50.93 | 51.28 |
| mCov | 56.49 | 60.73 | 61.73 | 61.82 |
| 正确归属覆盖率 | 69.35 | 78.71 | 77.58 | 77.88 |

无修复是 `abstain/final`：同帧同点多ID歧义弃权，保留原固定final后处理，不执行历史修复。main是 `10c6dd44` 的 partial＋E1＋swap/final，保留已记录的room0来源替换及room2人工例外。四方案逐景全部指标、累计TP/FP/FN、七景均值及来源锁见 [详细报告](experiments/online_v2/analysis/REPORT_with_no_repair.txt) 和 [机器可读结果](experiments/online_v2/analysis/comparison_with_no_repair.json)。

最后一次400帧修复本身使七景14条分支的AP50有6条上升、7条下降、1条不变。这与最终地图优于main不是同一个比较：最终地图已包含此前全部修复。

明确退化包括 office2 A/B的GT76与GT89瓶子混入纸巾实例，office0的GT22植物跌破0.5匹配阈值，以及office4 B的GT19椅子继续分裂。office0、office2的在线A/B最终AP50均低于无修复。不能只报告均值提升或删掉退化结果。逐GT诊断见 `experiments/online_replica7_window_AB_20261023/evidence/final/diagnostics.json`。

## 优化耗时与不能省略的限制

同一100帧修复起点继续101～200帧的一次串行对照：主建图3235.885→98.909秒（32.72倍），含恢复与导出3290.492→158.050秒（20.82倍）。不含候选、SAM2、E1、提交及评价；不是完整修复流程加速比，也不是多次平均。

固定同一表面和点序的控制计算精确一致，但独立真实闭环曾出现Open3D重复点顺序差异、Gate阈值翻转及2个最终标签差异，严格等价未通过。本分支保留该失败限制，未增加点排序或放宽验收。

main与无修复使用同一几何、GT及对应索引；在线与它们的几何数组/索引哈希不同（点数相同不证明逐元素相同）。四方案比较是同评价协议下的已有流程结果，不是固定同一表面的单因素消融，不能把全部增益归因于重建频率或仅窗户规则。room0不是与其余七景同历史的完整八景实验。

本地查看页面：`http://127.0.0.1:8801/replica7/index.html`（依赖本机静态服务与已下载PLY，不是公网部署）。自动实验跟进已暂停。
