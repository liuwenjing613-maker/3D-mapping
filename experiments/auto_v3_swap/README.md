# auto-v3-swap：八场景已验证流程冻结

本版本冻结 2026-10-20 完成的八场景新顺序实验，继承 auto-v2 提交 `2ac64f37ffd9f376ac1e3f4a40b2d333db01439a`。算法 `frozen=true`；评价仍为 main 默认 `object_observed_repair / revision 2`、`frozen=false`，固定 350 个可评价 GT。它是现有流程的完整源码及数据来源快照，不是无需数据即可运行的通用新场景工具。

## 固定流程

1. auto-v1：CropFormer/SAM 候选、语义来源路由、交集/过分割处理、完整 mask 过滤、组内种子选择；不裁剪种子、不重编号。
2. 背景筛选：原 mask 内高置信墙/地/顶/地毯占比 ≥90%，语义像素置信度 ≥0.7。
3. 共同可见 3D IoU ≥0.8 去重；使用固定主种子，不传递合并。
4. **完整支持包含先执行**：交集 / 较小支持严格 >70%，排除小种子。
5. **覆盖充分后执行**：来源帧 p≥0.95 且 r≥0.95，跳过修复。
6. 第五项筛选：p>0.1 且 r<0.2 排除；SegFormer top class 为 `windowpane` 或 `cabinet` 的种子仅豁免这一项。
7. 同一 mask 保留原始 track ID；来源帧 Dice 关联统一计算一次，关联旧 ID 或创建新 ID，后续追踪沿用同一 ID。仅整组成员、种子与配置完全一致的追踪案例可复用；组成员变化时整组重跑。
8. 默认修复是**原版 E1 whole**。partial 是单独保留的对照：仅将 whole 撤票计划改为既有 partial 方式，不改种子、追踪、ID、可信范围、门控与评价。原冻结 E1 数值内核不变。
9. main R2 对无修复、旧顺序 E1/partial、新顺序 E1/partial 使用相同 TSDF、GT 范围与对应索引；网页展示全流程、分组排除原因、最终 ID 决策及两类 PLY。

p=C/N，r=C/I；N 是种子来源帧可见的完整支持点数（含未分配点），I 是该帧与种子重叠最多的正实例的全部可见点数，C 是交集。

## 必须保留的已确认例外

- room0 G030/M0466 使用 CropFormer whole-frame，不能换回 SAM。源 mask 哈希写入 config.json。
- room2 G017、G025 是历史人工恢复，不属于窗户/柜子自动豁免。当前 room2 保留35张，不能把它描述为完全自动35张；不把该例外推广到其他场景。
- 八场景637个 auto-v1 主种子，最终196张：room0=35、room1=22、room2=35、office0=19、office1=22、office2=23、office3=22、office4=18。旧顺序201张，本次移除5张。

## 仓库内容与入口

`sources/experiments/` 保存服务器实际运行的逐字节源码，包括上游生成、筛选、追踪、E1、partial、调度及其第一方依赖。原 auto-v1/auto-v2 与原 E1 冻结树由父提交继承。`source_lock.json` 对全部新增源码、实测输入、清单与验证证据记录 SHA-256 和服务器来源；`runtime_lock.json` 记录外部模型/环境和依赖版本。`fixtures/<scene>` 保存原始实测计数、语义结果、原始及最终种子清单、决策、ID 关联；`evidence` 保存八场景指标和既有验收记录；`reporting` 保存页面构建与检查程序。大数据、模型、追踪结果、图片和 PLY 保留服务器，通过来源锁定位。

在仓库根目录执行，只读核验并重新计算八场景筛选：

```bash
python -B experiments/auto_v3_swap/verify.py
# 在原服务器同时核验部署源码、637张原始二值mask及外部模型文件：
python -B experiments/auto_v3_swap/verify.py --server-sources
python -m unified_eval.cli current-protocol
```

第三条评价命令必须在 `/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009` 执行。其评分源码基准为 `cd260569e89de1e9525f82c45ee10842b80616b0`，审核归档为 `d035f68770cefab27c9751e97062c72faa07367c`。不得替换为历史评价器。

## 重现实验的执行顺序与边界

快照刻意保留原服务器绝对路径，以保证逐字节一致。**不要直接重跑历史 launcher 写入已完成结果目录**。重新计算模型/E1 时，在独立 `/home/chenkejun/CVPR` 代码目录和 `/data/chenkejun/CVPR/results` 输出目录准备路径适配副本；记录适配差异，只允许改路径/调度，不改数值核、阈值、种子、ID或R2。源图、模型、TSDF与 mask 按来源锁验证。公开仓库不含私有服务器数据，克隆后可运行筛选重放；完整模型/E1重算需要服务器原数据和环境。

- room0 上游：`sources/experiments/auto_v2_room0_20261019/run_v1.py`，随后使用已确认来源修正及 `auto_v2_room0_pr35_20261019` 配置。
- room1/office0–4 上游：`sources/experiments/auto_v2_replica6_20261019/<scene>/run_v1.py`、`run_remaining.py`；包含 frontend、grouping、v1、v2、E1 原始代码。
- room2 上游：父提交 `experiments/auto_v1`；原始r4清单与测量见 room2 fixtures，保留两个人工恢复。
- 八场景新顺序：`sources/experiments/replica8_swap34_20261020/select_all.py`、`rules/`；由 `config.json` 锁定输入与复用关系。
- room1 E1/partial：`sources/experiments/room1_swap34_20261020`；office1/3/4：`sources/experiments/replica8_swap34_20261020/<scene>`；其他五场景经种子/帧/track ID/hash逐项核验后复用原结果。
- 每场景 E1 `run_selected.py` 先准备种子/核验，再追踪、统一ID关联、严格一票修复、历史核心与自适应扩展，最后调用 main R2。partial `controller.py` 先原版回放核验，再执行撤票对照和同口径评价。
- 八场景 `controller.py` 保存实际执行调度；office4曾独立提前启动，只有调度发生变化，见 `orchestration_update.json`。历史 room1 partial 缺来源锁的失败发生在映射前，成功版本和完整 provenance 已归档。
- `reporting` 中 collector/build_page/export/verification 程序用于重建报告。原结果位于 `/data/chenkejun/CVPR/results/replica8_swap34_20261020`，完整页面归档在其 `review` 下；图片/PLY来源及哈希见 `evidence/report_archive.json`。

## 验证结果

八场景筛选重放、637张种子来源、196张最终选择及ID保持、80条指标记录、350个GT口径和页面/PLY检查均有证据。八场景等权宏平均 F1/PQ/AP：无修复 0.627947/0.490819/0.187719；旧 E1 0.618690/0.490281/0.192645；新 E1 0.622679/0.496061/0.199189；旧 partial 0.618099/0.489955/0.192414；新 partial 0.624777/0.496920/0.199145。新顺序改善旧顺序，但宏平均F1仍低于无修复，不能宣称所有指标全面优于基线。

新冻结流程注册在 `instance_repair/configs/current_auto_pipeline.json`；原 `current_architecture.json` 继续指向未修改的E1内核。后续算法变化必须创建新版本和新来源锁，不覆盖本冻结快照。
