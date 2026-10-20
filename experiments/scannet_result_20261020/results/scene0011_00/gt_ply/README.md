# scene0011_00 GT 实例 PLY

两个文件均包含本轮评价的 27 个 GT 实例，保持原始 ScanNet 坐标，单位为米。
- `scene0011_00_gt_instances.ply`：这些实例的完整 GT 网格表面顶点，共 113,836 点。
- `scene0011_00_gt_instances_observed.ply`：严格对应 R2 评价中的可见目标区域，共 78,514 点。

完整表面版包含这些目标实例的未观测区域，用于查看形状，不是扩大评价范围；不包含墙、地面等本轮非目标实例。两个文件都是顶点点云，不含三角面。

配色优先匹配严格一票无修复基线，再为剩余 GT 匹配 E1、partial。沿用已有预测 PLY 的精确 RGB；同一个预测颜色只分配给一个 GT，避免合并错误掩盖两个不同物体。匹配以已冻结评价重叠矩阵的 IoU 作一对一优化，门槛为 IoU≥0.05、交集≥10 点、GT 覆盖率≥0.05。这里只建立可视化对应，不等同于正确预测；IoU<0.25 的对应标记为弱重叠。

颜色来源：基线 22 个，E1 3 个，partial 0 个，独立新颜色 2 个。弱重叠对应 8 个。

PLY 的 `instance_id` 是 GT 原始 ID（官方 objectId+1），`color_prediction_id` 是借用颜色的预测 ID（-1 表示新颜色）；二者不能混用。`evaluation_target` 标记是否参与本轮评价，`reference_vertex_index` 可回查原始 GT。

完整对应、三个方法各自最佳重叠实例、颜色竞争和输入哈希见 `color_mapping.json`。原预测、GT、评估结果与旧 PLY 不变。

| GT ID | 类别 | 颜色来源 | 预测 ID | RGB | IoU | 注记 |
|---:|---|---|---:|---|---:|---|
| 2 | kitchen cabinets | baseline | 40 | 131,79,239 | 0.4451 | 基线最佳颜色有合并竞争 |
| 3 | table | baseline | 7 | 85,239,79 | 0.3941 |  |
| 4 | armchair | baseline | 767 | 239,110,79 | 0.0708 | 弱重叠 |
| 5 | chair | baseline | 25 | 79,239,192 | 0.3605 | 基线最佳颜色有合并竞争 |
| 6 | chair | E1 | 1118 | 239,79,115 | 0.0680 | 弱重叠；基线最佳颜色有合并竞争 |
| 7 | chair | baseline | 8 | 239,79,132 | 0.5283 |  |
| 8 | chair | baseline | 5 | 239,165,79 | 0.5574 |  |
| 9 | chair | baseline | 6 | 119,79,239 | 0.5219 |  |
| 10 | chair | baseline | 1 | 79,125,239 | 0.6376 |  |
| 11 | chair | baseline | 2 | 172,239,79 | 0.3974 |  |
| 12 | chair | baseline | 4 | 79,239,212 | 0.4781 |  |
| 13 | chair | baseline | 245 | 79,239,161 | 0.5703 |  |
| 14 | chair | baseline | 253 | 79,239,107 | 0.1258 | 弱重叠 |
| 16 | trash can | baseline | 455 | 202,239,79 | 0.6503 |  |
| 18 | table | baseline | 337 | 132,239,79 | 0.2825 |  |
| 19 | tv | baseline | 362 | 138,79,239 | 0.4277 |  |
| 20 | refrigerator | new_distinct_color | -1 | 17,178,44 | — | 没有可用的独立匹配颜色；基线最佳颜色有合并竞争 |
| 21 | cabinet | baseline | 287 | 79,239,119 | 0.7042 |  |
| 24 | window | baseline | 300 | 79,239,153 | 0.2427 | 弱重叠 |
| 25 | window | baseline | 10 | 226,239,79 | 0.3184 | 基线最佳颜色有合并竞争 |
| 26 | kitchen counter | E1 | 1073 | 239,224,79 | 0.1181 | 弱重叠；基线最佳颜色有合并竞争 |
| 27 | sink | new_distinct_color | -1 | 37,17,178 | — | 没有可用的独立匹配颜色；基线最佳颜色有合并竞争 |
| 28 | refrigerator | baseline | 46 | 79,239,171 | 0.6319 | 基线最佳颜色有合并竞争 |
| 29 | cabinet | E1 | 1048 | 110,79,239 | 0.0901 | 弱重叠；基线最佳颜色有合并竞争 |
| 31 | doors | baseline | 713 | 79,87,239 | 0.1472 | 弱重叠 |
| 32 | doorframe | baseline | 23 | 193,239,79 | 0.0925 | 弱重叠 |
| 33 | trash can | baseline | 743 | 208,239,79 | 0.5635 |  |
