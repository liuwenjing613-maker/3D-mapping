# revision 2 提升为 main 当前开发基准（2026-10-09）

用户已明确指定后续统一使用 `object_observed_repair / revision 2`，配置为 `unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json`。

- 评分来源：`cd260569e89de1e9525f82c45ee10842b80616b0`。
- 审核归档：`d035f68770cefab27c9751e97062c72faa07367c`。
- GT：八场景固定 350 个可评价对象；6 个极小对象继续待审，168 个 UNKNOWN 继续保留。
- 用途：固定 TSDF 的实例地图质量与成对标签修复。
- 状态：当前开发评价基准，`frozen=false`；提升到 main 不等于论文正式冻结。

本次保留 audit_r2.json 的原始字节，SHA256 为 `6668792cbbeff3b6a6b6f2302ad8baa1b202f463ae26306350d4617c07f601a0`。七个评分核心文件与 cd260569 的原始字节完全相同；原 revision 1 开发锁及 revision 2 审核归档也不修改。

| 改动 | 作用 |
|---|---|
| configs/current_protocol.json | 唯一当前选择：记录配置、来源提交、评分源码哈希和逐场景 GT 文件/范围/观测哈希 |
| current_protocol.py | 校验当前配置、评分源码和固定 GT；只有显式历史复现才能使用其他配置，当前配置即使带历史标记也继续受锁保护 |
| cli.py | 默认 revision 2；新增 current-protocol、adapt-surface、eval-repair-pair；原 eval-scene/eval-batch 明确记录 profile 与 revision |
| adapt_instance_map_replica_ca.py | P1 适配入口默认当前配置，并验证固定 GT；旧协议须显式历史复现 |
| run_object_observed_repair_development.py | 原 GT 构建/验收 runner 标为历史入口，显式历史标记才能运行；后续使用锁定 GT |
| AGENTS.md、README、profile 文档 | 明确当前唯一基准、使用命令和历史结果边界，清除 README 的“旧报告是最新协议”表述 |
| test_current_protocol.py | 12 项防混用验收：默认选择、配置与源码修改、GT 修改、历史标记和在线前缀范围 |

服务器 103 项测试全部通过，零 skip/failure/error。八份固定 GT 文件逐项 SHA256 一致。通过新 CLI 默认入口重新评分 room0 P1 补洞、比较原始→补洞成对修复，并再次适配完整原生表面；数值逐项与 d035f687 审核归档一致，原对应缓存未改变。完整证据见本目录 XML 与 JSON。

新命令不传 --config 即使用 audit_r2.json。可用 `python -m unified_eval.cli current-protocol` 查看当前来源锁；实际例子见仓库 README 开头。历史复现需 `--historical-protocol` 且单独存放结果；最终 400 帧的 350-GT 范围不用于在线前缀评价。

main 采用原审核分支及本次默认入口提交，不使用强制推送。原审核报告的“独立审核分支、未合入 main”描述属于当时的历史记录，当前使用状态以 current_protocol.json 和 README 为准。
