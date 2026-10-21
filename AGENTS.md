# CVPR 当前开发评价基准

- 后续统一使用 `unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json`，`object_observed_repair / revision 2`。当前选择与来源锁见 `unified_eval/configs/current_protocol.json`。
- 评分源码固定为 `cd260569e89de1e9525f82c45ee10842b80616b0`；审核归档固定为 `d035f68770cefab27c9751e97062c72faa07367c`。更新 main 的入口和说明不改变这两个来源提交的含义。
- GT 为八场景共 350 个固定可评价对象。6 个极小 GT 继续待审，168 个 UNKNOWN 保持 UNKNOWN。不重新生成或扩大当前评价范围。
- 用于固定 TSDF 上的实例地图质量与成对标签修复。成对修复必须共用相同几何、GT 范围和对应索引；最终 400 帧范围不能用于在线前缀评价。
- 当前入口：`python -m unified_eval.cli current-protocol`、`adapt-surface`、`eval-scene`、`eval-batch`、`eval-repair-pair`。默认 revision 2；旧协议复现须显式使用 `--historical-protocol`，结果单独标记，不能混入当前比较。
- 配置仍为开发基准、`frozen=false`。用户采用当前开发基准不等于批准论文正式冻结，不擅自修改参数、评分源码或这一标记。
- 服务器代码、配置、文档在 `/home/chenkejun/CVPR`；数据和结果在 `/data/chenkejun/CVPR`。新的评价使用独立输出目录，不覆盖已锁定的开发和审核归档。

## 冻结 E1 数值内核

- E1 数值内核为 `instance_repair/configs/current_architecture.json` 指定的 `P1-A1-strict-vote-E1-20261012`，说明见 `docs/FROZEN_E1.md`；当前完整算法基线由下面的 current_auto_pipeline.json 注册。
- E1 使用自适应可信范围扩展、历史观察重关联、严格一票歧义弃权；不启用 E2 的局部新 Mask 约束。
- `instance_repair/frozen_e1_20261012` 和其源码锁为冻结记录。后续算法开发使用新版本/新输出，不直接覆盖冻结实现及参考结果。用户明确要求升级冻结版本时，应重新验证并更新注册。
- 算法 `frozen=true` 不改变评价协议 `frozen=false`。评价继续从 main 的 unified_eval 默认 R2 入口执行。

## 当前冻结算法基线：partial＋E1＋swap（2026-10-21）

- 当前默认基线为 `partial-e1-swap-20261021`，由 `instance_repair/configs/current_auto_pipeline.json` 注册；配置、八场景结果锁及验证入口见 `experiments/partial_e1_swap_baseline_20261021/README.md`。
- 自动种子沿用 auto-v3-swap：637个原始种子、196个保留种子；完整支持包含先于覆盖充分筛选，窗户/柜子仅豁免第五步。保留 room0 M0466 来源修正及 room2 G017/G025 历史人工例外。
- 默认撤换方式为既有 partial，随后执行原核心历史重关联及 E1 自适应历史扩展；最终结果为 `whole_to_partial` 的 final。它不是 onebyone，也不包含新保护规则。原 E1 数值内核及 auto-v3-swap whole 历史归档不变。
- 修改本基线须建立新版本和独立输出，不覆盖源码锁与结果；算法 frozen=true，评价仍为 main R2、frozen=false。不得将旧归档 config 的 original_E1_whole 当作当前默认。
