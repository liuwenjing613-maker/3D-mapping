# main 双向覆盖 0.90（2026-11-01）

当前 main 的第四步跳过修复条件为 **p ≥ 0.90 且 r ≥ 0.90**，等号包含在内。p=C/N、r=C/I；N 为种子支持点数（含未分配点），I 为来源帧重叠最多的正实例的全部可见点数，C 为交集。必须匹配到正实例；只满足一侧，或只有 Dice ≥0.90，均不跳过。实现使用整数比较 `10*C >= 9*N and 10*C >= 9*I`。

默认注册见 [current_auto_pipeline.json](../../instance_repair/configs/current_auto_pipeline.json)，执行实现为 [seed_filter_rules.py](../../instance_repair/seed_filter_rules.py)。它从冻结 auto-v3-swap 规则派生，唯一数值变化是双向 0.95→0.90；旧规则、源码锁和历史输出不覆盖。

## 接入当前流程

部署适配器在完成全触边筛选、重新分组和语义/v2测量后，将原 `from filter_rules import run_filters` 改为：

```python
from instance_repair.seed_filter_rules import run_filters
selection = run_filters(measurements)
```

随后沿用既有 `apply_semantic_exemption` 与按 Mask 哈希/帧绑定的历史例外处理、清单导出及 [validate-selected](../seed_boundary_20261101/README.md)。语义豁免只作用于第五步，不能恢复第四步跳过的种子。规则路径和豁免实现记录在 config.json。不要直接运行旧归档 launcher；新实验在独立目录适配入口。

完整支持包含仍先于覆盖充分筛选；ID 继承使用原关联规则，后续 partial＋E1、main R2 和整场景结束执行一次 B 均保持不变。[全触边策略](../seed_boundary_20261101/README.md)继续生效。

```bash
python -B -m unittest discover -s instance_repair/tests -p 'test_seed*.py' -v
```

## 已有验证范围

服务器 `/data/chenkejun/CVPR/results/main_skip90_repair_20261101` 已完成原增量 TSDF 的八场景离线修复：可投影种子 163→124，98 批追踪复用，3 批重算，office1 整场复用。固定相同 TSDF、350 GT 和 R2 对应索引，双方最终 B 后 pooled F1 为 59.46%→60.33%、AP50 为 35.19%→36.44%、PQ 为 46.33%→47.07%；office2 F1 下降 2.22 个百分点。这些是开发评价结果。

上述实验使用原 crop-only 候选；不代表当前全触边规则与 0.90 组合已重跑。代码验证另外用同一八场景实测计数重放新规则，检查与已完成 0.90 实验的阶段决策、保留种子和记录一致。来源与文件哈希见 source_lock.json。

截图中的 office0 G020/M0094（frame 1315）为 p=11424/12692≈0.90009、r=11424/11448≈0.99790：旧 0.95 不跳过，新 0.90 跳过。
