# scene0608_00 GT 实例 PLY

两个文件均包含本轮评价的 34 个 GT 实例，保持原始 ScanNet 坐标，单位为米。
- `scene0608_00_gt_instances.ply`：这些实例的完整 GT 网格表面顶点，共 75,865 点。
- `scene0608_00_gt_instances_observed.ply`：严格对应 R2 评价中的可见目标区域，共 54,224 点。

完整表面版包含这些目标实例的未观测区域，用于查看形状，不是扩大评价范围；不包含墙、地面等本轮非目标实例。两个文件都是顶点点云，不含三角面。

配色优先匹配严格一票无修复基线，再为剩余 GT 匹配 E1、partial。沿用已有预测 PLY 的精确 RGB；同一个预测颜色只分配给一个 GT，避免合并错误掩盖两个不同物体。匹配以已冻结评价重叠矩阵的 IoU 作一对一优化，门槛为 IoU≥0.05、交集≥10 点、GT 覆盖率≥0.05。这里只建立可视化对应，不等同于正确预测；IoU<0.25 的对应标记为弱重叠。

颜色来源：基线 33 个，E1 1 个，partial 0 个，独立新颜色 0 个。弱重叠对应 2 个。

PLY 的 `instance_id` 是 GT 原始 ID（官方 objectId+1），`color_prediction_id` 是借用颜色的预测 ID（-1 表示新颜色）；二者不能混用。`evaluation_target` 标记是否参与本轮评价，`reference_vertex_index` 可回查原始 GT。

完整对应、三个方法各自最佳重叠实例、颜色竞争和输入哈希见 `color_mapping.json`。原预测、GT、评估结果与旧 PLY 不变。

| GT ID | 类别 | 颜色来源 | 预测 ID | RGB | IoU | 注记 |
|---:|---|---|---:|---|---:|---|
| 1 | guitar | baseline | 12 | 79,239,159 | 0.6843 |  |
| 3 | table | baseline | 267 | 239,93,79 | 0.6392 |  |
| 4 | keyboard piano | baseline | 1025 | 79,239,225 | 0.6261 |  |
| 5 | plant | baseline | 2 | 172,239,79 | 0.5416 |  |
| 6 | lamp | baseline | 600 | 227,79,239 | 0.7169 |  |
| 7 | coffee table | baseline | 18 | 239,199,79 | 0.7595 |  |
| 9 | ottoman | baseline | 336 | 79,86,239 | 0.4978 |  |
| 10 | ottoman | baseline | 391 | 79,93,239 | 0.6243 |  |
| 11 | tv stand | baseline | 5 | 239,165,79 | 0.3830 |  |
| 13 | door | baseline | 316 | 112,239,79 | 0.5634 |  |
| 14 | picture | baseline | 223 | 228,79,239 | 0.7570 |  |
| 15 | plant | E1 | 119 | 79,195,239 | 0.0562 | 弱重叠；基线最佳颜色有合并竞争 |
| 17 | curtain | baseline | 8 | 239,79,132 | 0.1338 | 弱重叠 |
| 18 | window | baseline | 7 | 85,239,79 | 0.3381 |  |
| 19 | plant | baseline | 1087 | 210,79,239 | 0.5297 |  |
| 21 | clock | baseline | 1358 | 120,239,79 | 0.4466 |  |
| 22 | picture | baseline | 689 | 232,79,239 | 0.6529 |  |
| 23 | picture | baseline | 676 | 198,79,239 | 0.5455 |  |
| 24 | end table | baseline | 610 | 239,79,79 | 0.3214 |  |
| 26 | pillow | baseline | 244 | 207,79,239 | 0.5609 |  |
| 27 | couch | baseline | 209 | 237,239,79 | 0.6749 |  |
| 28 | pillow | baseline | 298 | 232,239,79 | 0.6741 |  |
| 29 | sofa chair | baseline | 455 | 202,239,79 | 0.7091 |  |
| 30 | sofa chair | baseline | 377 | 239,79,80 | 0.4381 |  |
| 33 | picture | baseline | 683 | 239,192,79 | 0.3840 |  |
| 34 | picture | baseline | 690 | 79,239,185 | 0.6269 |  |
| 35 | picture | baseline | 674 | 79,186,239 | 0.5833 |  |
| 36 | picture | baseline | 673 | 239,79,139 | 0.7155 |  |
| 37 | picture | baseline | 720 | 239,79,94 | 0.5281 |  |
| 38 | picture | baseline | 721 | 79,140,239 | 0.5843 |  |
| 39 | picture | baseline | 702 | 239,79,214 | 0.4079 |  |
| 40 | picture | baseline | 677 | 79,239,152 | 0.4808 |  |
| 41 | picture | baseline | 688 | 200,239,79 | 0.4624 |  |
| 44 | tv | baseline | 3 | 239,79,219 | 0.3623 | 基线最佳颜色有合并竞争 |
