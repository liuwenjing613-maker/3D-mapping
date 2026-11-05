# two-audit 当前处理说明（2026-11-05）
状态：暂停后续筛选，供人工审查。本分支不是最终修复地图。

## 数据流
1. 输入：Replica 8 场景已有全深度增量 TSDF＋DBSCAN，400 个处理帧（原始帧 0,5,...,1995）；不重新运行分割模型，不修改地图。
2. 投影缓存：原始 Mask 和深度、位姿投影到固定 TSDF 表面点索引。生成每帧可见集合 V 与各 Mask 支持集合 S。深度容差 0.015m。
3. 稳定簇：跨帧 Mask 与固定首个锚点比较共视 IoU；共同支持至少25点，IoU至少0.55。帧内唯一匹配，歧义暂缓；至少3帧、2个独立视角（0.1m或10度）为稳定簇。稳定簇不是最终地图实例。
4. 每稳定簇取支持点最多的原始 Mask；并列较早帧。先选后筛，失败不补选。筛选支持点按原流程0.02m投影重新计算，不能与选种0.015m支持混同。
5. 复用原筛选：全触边排除→噪声→普通3D IoU>0.25固定主种子分组→背景语义→共视IoU>=0.8去重→较小支持被覆盖>70%的包含筛选→P/R均>=0.9无需修复排除→P>0.1且R<0.2排除（windowpane/cabinet本步豁免）→投影预检→最终ABC。最终ABC仅删除合格大A，windowpane/blind保护；没有执行修复。
6. 对比两批最终主种子：原full-depth main final_abc清单，与本次稳定簇选种后最终清单。普通3D IoU>0.25、支持点降序固定主种子分组，不做链式合并；分组保留所有Mask。

## 当前结果
|场景|原本最终主种子|本次最终主种子|
|---|---:|---:|
|room0|21|19|
|room1|9|4|
|room2|19|10|
|office0|6|待包含规则确认|
|office1|10|待包含规则确认|
|office2|13|12|
|office3|10|待包含规则确认|
|office4|9|7|

office0、office1、office3原包含规则抛出 Chained qualifying containment needs user review，未绕过、未形成最终清单。用户选择先看Mask，尚未批准淘汰。
office0：A053/M000493（帧1375，68点）→A033/M000150（帧405，383点）覆盖82.35%；后者→A013/M000149（帧390，2757点）83.81%。
office1：A029/M000067（帧220，82点）→A017/M000093（帧780，507点）100%；后者→A008/M000084（帧580，1827点）75.74%。
箭头仅表示直接的较小支持覆盖关系，不推出首尾包含。

## 代码入口
- instance_repair/two_audit_sources.py、two_audit_projection.py、two_audit_cache.py：原始来源、投影缓存。
- instance_repair/two_audit_matching.py、two_audit_tracks.py：跨帧稳定簇。
- scripts/two_audit_seed_filter.py：选种及完整筛选，调用服务器归档原筛选代码。
- scripts/build_two_audit_seed_review.py：每步筛选记录页面。
- scripts/build_two_audit_original_seeds.py：原本最终主种子导出。
- scripts/build_two_audit_combined_seeds.py：两批最终种子的固定主种子分组。
- scripts/build_two_audit_replica8_final.py：八场景最终Mask页面及来源校验。
- scripts/build_two_audit_containment_review.py：两条待确认包含链的六张原始Mask。
- scripts/run_two_audit_remaining_six.py：六场景批处理；不自动裁决包含链。
- scripts/benchmark_two_audit.py：用户取消的性能测试，不存在完整新实测结论。

## 服务器结果与依赖
代码：/home/chenkejun/CVPR/worktrees/two-audit
结果根：/data/chenkejun/CVPR/results/two-audit/replica_room0_office2_v1_20261104
原数据：/data/chenkejun/CVPR/results/unified_full_depth_20261101/main
原筛选实现：/home/chenkejun/CVPR/experiments/unified_full_depth_20261101/main/<scene>/selection_code/400/A
最终结果：结果根/stable_seed_filter_v1/<scene>/final_abc/seed_manifest.json（仅完成场景）
页面：结果根/stable_seed_filter_v1/review/replica8
各阶段保留输入哈希、manifest、筛选证据及校验记录。数据、权重、点云和图像未加入Git；代码依赖上述服务器归档，不能仅克隆后脱离数据直接运行。
页面/seeds与/clusters分别是种子检查和观测簇检查，都不是修复后的最终实例地图。
本轮没有运行GT质量评价，没有改变固定R2协议，没有运行地图修复。
