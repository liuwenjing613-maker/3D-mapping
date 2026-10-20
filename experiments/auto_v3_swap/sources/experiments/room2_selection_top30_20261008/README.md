# room2 前30人工标注页

本轮准备原始 P1-A1 room2 的 CONFLICT 排名前30区域（ROI-C0001～ROI-C0030）。按原生冲突点数降序，原有 component ID 决定同分顺序；保留90个原有固定视角，未使用GT，未改变排序、地图或旧标注。

## 页面

http://127.0.0.1:8785/room2_selection_top30_20261008/index.html

三栏显示原始整帧CropFormer缓存、局部CropFormer、同局部输入的SAM2.1。沿用room0的冲突投影、每mask独立显隐/高亮/只看、同步缩放、多个目标选择与允许触边保存。覆盖全部目标点、相关完整旧mask及15%上下文，最小400×226；宽上下文备选。大区域增设600×380局部窗口，明确标出部分覆盖，不拼接mask。

旧5例的75套候选保留相同视角、裁剪及逐像素ID；PNG重新编码不改变像素。旧标注仅供参考，可显式载入草稿，保存确认后计入本轮30例。正式storage key为`cvpr-room2-top30-selection-20261008-v1`；测试加`?qa=1`使用独立的`-qa`存储，导出文件及记录标记`is_qa:true`，不用于修复。

导出`room2_前30_修复候选选择_20261008.json`，包含原始排名、ROI UID、原图帧、裁剪坐标、模型、mask ID、触边ID及源数据hash。数据保存在新目录，旧room0/room2标注及页面保留。

## 运行和核验

服务器代码：`/home/chenkejun/CVPR/experiments/room2_selection_top30_20261008`

服务器数据：`/data/chenkejun/CVPR/revisable_instance_map/room2_selection_top30_20261008`

本地页面：`results/固定案例_三模型对比_20261006/room2_selection_top30_20261008`

`prepare_inputs.py`保存原始排名、30个核心/上下文点索引、RGB及状态投影的校验；保留旧5例的候选范围，其他25例采用room0相同策略。仅同一PNG输入、相同模型代码/参数的结果可复用。

`infer_cropformer.py`与`infer_sam.py`在GPU0/1生成缺少的候选。315项含314套候选及1个整帧控制；CropFormer控制与原缓存逐像素一致。SAM2.1 raw proposals仍存档。

`render_annotation.py`生成942张可选择的ID图和对应状态投影。`verify_sources.sh`核对room0模型权重/配置/实现/分配规则及冻结源未变。`verify_candidates.py`核对每份资源hash、尺寸、ID位置、像素数量、投影直方图/锚点及旧5例的逐像素一致性。`build_selection_page.py`在本地复用room0界面并冻结新导出字段。

网页功能测试只写独立QA记录，覆盖模型选择、触边保存、逐mask显隐、视角和裁剪切换、旧草稿、刷新恢复、30例完成进度与实际JSON下载。

本轮仅准备人工标注候选，没有SAM视频追踪或TSDF换票。收到人工导出后，再检查重复物体、双向SAM2.1追踪、联合撤回/补票，展示3D变化及同参数统一v3评估。
