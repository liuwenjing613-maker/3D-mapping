# 全触边排除：2026-11-01

当前种子规则为 `full-frame-and-crop-edge-20261101`。所有来源模型统一执行：原图四边任意前景像素触边，或任一来源别名的人工裁剪框边缘被前景触及，排除整张 Mask。边界用原始 1200×680 分辨率判断，裁剪坐标为左闭右开；不腐蚀、不裁掉边缘像素、不改采样或投影阈值。model 0 是整图来源，model 1/2 是裁剪来源。

**执行顺序必须是：原有来源/语义路由 → 全触边排除 → 原有噪声过滤 → 重新分组与主种子选择 → 原有语义及 v2 筛选 → 最终触边检查 → 跟踪与修复。** 所有候选、主种子和辅助 Mask 均受约束；窗口、柜子和历史人工例外不能绕过触边检查。必须从 grouping_r2 的完整来源合格候选重新筛选，不能仅删除最终清单中的触边主种子。

## 可执行入口

在仓库根目录使用当前分支对应的原有 v1 规则目录。输出必须是不存在的新目录：

```bash
python -B -m instance_repair.seed_boundary current-policy
python -B -m instance_repair.seed_boundary screen \
  --grouping /path/to/grouping_r2/grouping.json \
  --supports /path/to/grouping_r2/surface_supports.npz \
  --pairs /path/to/grouping_r2/pair_metrics.npz \
  --rules-dir /path/to/original/v1 \
  --out /path/to/new-run/v1
```

`screen` 输出新的 v1 grouping、完整 Mask 来源引用、supports、pair_metrics 和逐 Mask 边界审计；保留来源 ID 和 Mask 哈希，重新生成分组 ID。它调用传入的原有噪声、固定主种子分组与来源约束实现；3D IoU 仍严格 >0.25。输出不是最终选中清单。部署适配器必须把后续语义分类和 v2 测量的输入改为这个新 v1 目录，重新计算筛选，不能复用旧 group ID 的语义或最终选择结果。历史例外只能按原 Mask 哈希及帧绑定，并且必须通过边界检查。

在跟踪前，使用 v2 最终清单与新的完整候选 grouping 强制检查：

```bash
python -B -m instance_repair.seed_boundary validate-selected \
  --manifest /path/to/new-run/v2/seed_manifest.json \
  --grouping /path/to/new-run/v1/grouping.json \
  --mask-root /path/to/new-run/v1
python -B -m unittest discover -s instance_repair/tests -p 'test_seed_boundary.py' -v
```

`validate-selected` 拒绝尚未完成后续筛选的 v1 清单，检查主种子与辅助 Mask 的来源、实际像素和哈希。不允许只信任 `touches_frame=false` 等缓存标记。检查失败必须停止跟踪/修复。

## 版本和实验状态

main、auto-v3-swap、online-v1、online-v2 共用本规则与验证入口。main/auto-v3-swap 的新默认配置要求补上原图边缘排除；在线版本原本已有全触边排除，数值行为应保持一致。在线 A/B 的候选时间范围、历史继承、来源路由和 checkpoint 策略不变。

原冻结源码、原 E1 数值内核、原评分器及历史结果不覆盖。更新 `current_auto_pipeline.json` 表示新种子策略已登记，**不表示八场景地图已重跑或新指标已产生**。旧冻结包按原提交 checkout 后运行历史验证；不要把旧包对旧注册文件的哈希检查当成新策略验证入口。新入口如上。

2026-11-01 OVIMAP 对齐实验的已交付 main 仍是 crop-only：最终163个种子中58个触原图边缘，room0为9个；ID14的131像素 Mask 触左边和下边。该证据用于回归，不代表全触边重新分组后的最终数量。修正规则后的地图和评分需在独立实验目录重跑；本次 GitHub 同步不热替换正在运行的实验快照。原生 OVIMAP 几何缺失是另一个问题，本改动不增加或删除 TSDF 几何。

当前 main 的后续 v2 筛选必须使用 `instance_repair.seed_filter_rules.run_filters`（双向 ≥0.90），接入和验证见 [skip90 版本](../seed_skip90_20261101/README.md)。归档 filter_rules.py 的 0.95 仅供历史复现。
