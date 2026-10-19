# 42 个种子的 p / r 筛选

2026-10-19，用户确认“按现有 42 个继续筛选”。输入是完整 TSDF 包含率筛选后保留的 42 个，之前已排除的 G080、G081、G085 不重新纳入。

只增加用户指定条件：**p > 10% 且 r < 20% 时排除**。沿用 B 的单帧结果：C 为种子与最大重合正实例的交集点数，N 为种子的完整支持点数（含未分配点），I 为该实例在来源帧可见的全部点数。p=C/N、r=C/I。使用整数条件 `10*C>N and 5*C<I`，等于边界时保留，不引入其他条件。

基准仍是 B 使用的严格一票未修复地图。原始 p/r、输入清单、种子 Mask、图像缓存都通过 SHA-256 锁定；不重新投影，不执行分割模型，不生成分割图，不改 Mask 像素，不运行 E1，不使用 GT 或改变评价协议。

本轮排除 9 个：G017、G025、G069、G071、G074、G075、G077、G082、G086；继续保留 33 个。保留 G038、G084 等 p≤10% 的种子，即使 r<20% 也不排除。规则用于本次实验，筛选决定本身不构成实例正确性的真值标注。

服务器代码：`/home/chenkejun/CVPR/experiments/seed_pr_filter_20261019`；结果：`/data/chenkejun/CVPR/results/seed_pr_filter_20261019`。本地页面：`http://127.0.0.1:8786/room2_seed_pr_filter_20261019/index.html`。页面支持两组切换、仅种子 Mask、来源的全部分割 Mask、色块和 RGB，并展示 p/r 原始点数；全部图像引用既有页面缓存，单独搬动本页文件夹时应同时提供被引用的两套缓存。

`config.json` 保存输入路径、哈希与规则；`rules.py` 执行整数判断；`test_rules.py` 验证严格边界、AND 条件、无实例和四舍五入；`run.py` 导出新清单、复用图像清单和验证记录，并打包页面。运行：`python -B -m unittest discover -s . -p test_rules.py`，然后 `python -B run.py`。结果已经冻结时拒绝重复覆盖。

主要输出：`classification.json`（42 个完整判断）、`selection_binary.json`（9/33 两组）、`seed_manifest.json`（33 个原始 Mask）、`review_data.json`、`cached_images.json`、`verification.json` 和交付包。既有 B 和包含率实验结果保持原样。
