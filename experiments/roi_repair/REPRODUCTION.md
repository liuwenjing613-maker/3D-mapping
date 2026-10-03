# 运行环境与复现记录

本目录是已经执行完成实验的归档。脚本按实际运行版本保留，含服务器和本地绝对路径。模型推理、ROI 提取与状态证据计算在服务器运行；中文 PNG 和交互可视化在 Windows 本地生成。

## 服务器目录

| 用途 | 路径 |
|---|---|
| ROI 代码 | `/home/chenkejun/CVPR/experiments/p1a1_room0_roi_review_20261002` |
| ROI 结果 | `/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002` |
| CropFormer 局部实验代码 | `/home/chenkejun/CVPR/experiments/p1a1_roi_cropformer_zoom_20261003` |
| CropFormer 局部输入、标签、日志、结果 | `/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003` |
| P1-A1 原生证据和最终地图 | `/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002` |
| 地图原始场景配置与 GT | `/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0` |
| 模型权重 | `/data/chenkejun/CVPR/models/CropFormer_hornet_3x_03823a.pth` |
| 高清逐裁剪完整对照归档 | `/data/chenkejun/CVPR/revisable_instance_map/p1a1_roi_cropformer_zoom_20261003/room0_32ROI_all_comparisons.zip` |

旧环境路径为既有兼容入口，不在旧位置复制数据或模型。代码/配置在 `/home/chenkejun/CVPR`，新实验数据/日志/权重在 `/data/chenkejun/CVPR`。

## 已运行步骤

1. `detect_rois.py`：读取原生证据、最终实例标签和既有表面法线，生成表面图、核心/上下文/排序邻域索引、候选和检测清单。
2. `select_history.py`：读取历史 RGB-D 相机轨迹，检查 400 帧可见性，选三个视角，生成最终 `roi_ranked.json` 和 CSV。
3. `render_roi_assets.py`：选择 32 个展示 ROI，投影状态和邻近实例，输出 RGB 裁剪、PNG 及只读点数组。脚本保留历史 PLY 辅助函数，但当前主流程不调用；本分支不包含任何 PLY 输出。
4. `analyse_conflicts.py`：回查 23 个代表点的真实帧投票和竞争 mask；`fix_gt_names.py` 修正 GT reduced+1 类别编码（0 为未标注）；`render_annotated_cases.py` 等生成 13 个诊断案例。GT 只在此作为事后参考。
5. `prepare_inputs.py`：准备 96 个紧裁剪、39 个宽裁剪和对应整帧控制；`prepare_witness_inputs.py` 增加 24 个实际竞争来源裁剪和新增整帧控制。裁剪由原历史 RGB 提取，不含可视化标记。
6. `infer_crops.py`：固定 CropFormer 配置，运行 159 个局部输入和 79 个唯一整帧控制；逐输入记录 SHA256、标签、得分和控制标签一致性。已有输出经核验后跳过。
7. `check_witness_pixels.py`：对 24 个固定原生来源像素检查 F→L 变化；`render_zoom_comparison.py` 生成全部对照图和紧凑交互数据；`finish_zoom_review.py` 核验原始 RGB/标签与展示导出一致。
8. `verify_frozen_maps.sh`：确认原生证据和最终地图未改变。浏览器检查覆盖全部 159 组展示、32 个 ROI、宽/紧裁剪、证据来源视角、像素选择和移动布局。

`prepare_inputs.py` 和追加证据输入脚本对已经存在的输入有保护性断言，不应直接覆盖当前已完成实验。若需要新实验，先修改脚本里的输入/输出目录常量到独立 `/home/chenkejun/CVPR` 代码目录与 `/data/chenkejun/CVPR` 数据目录，并保持源数据、模型及配置 SHA256 匹配。

## 环境

- ROI/证据分析：`/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python`，使用 NumPy、SciPy、Pillow、plyfile。
- CropFormer：`/data/chenkejun/ovimap_runtime_20260908/envs/cropformer/bin/python`，使用已配置的 PyTorch、Detectron2 和 CropFormer。配置 `configs/entityv2/entity_segmentation/cropformer_hornet_3x.yaml`；完整展开配置与权重/预测器校验见 `cropformer_zoom/evidence/model_config.yaml` 和 `model_provenance.json`。
- 实际成功运行用 GPU1（RTX 5880 Ada），`OMP_NUM_THREADS=4`，随机种子 0。`start_native_inference.sh` 记录运行入口；推理日志为 `inference_native.log` 与后续 `inference_witness.log`。
- 原服务器驱动运行库通过任务目录下符号链接匹配已有 580.126.09 版本。`prepare_driver_links.sh` 记录该环境处理；没有更改系统驱动。不要把此硬件路径当作通用依赖。
- PNG 排版脚本依赖 Windows 微软雅黑 `C:/Windows/Fonts/msyh.ttc`；交互脚本保留生成时本地路径。迁移时需调整路径与字体。
- `visualizations/*.html` 是 Codex 内联 HTML 片段，不是带完整 document 外壳的独立网页；JSON、RGB 和标签已嵌入，查看内容不请求远程数据。

## 输入定位与校验

原生证据：`native/room0/P1-A1/surface_p0/surface_evidence.npz`，SHA256 `6066ad83c20b0120541f13d96033eecad172357459f5453389a8bb63d84c83ce`。

最终实例地图：`final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz`，SHA256 `156755555eec78c2f7177f4a2be7602466e8ab9d1d4c020fb99628054a997666`。

模型权重 SHA256：`71915b7e9dc63fc662b32c59dff513530a06221c3b22f6de7ff4d6ef82df1128`。

相机、原历史 RGB、冻结 mask 来源由原始 `raw.json` 定位；具体原生帧号、裁剪坐标及每个输入的 SHA256 在 `input_manifest.json`。该清单保留原服务器绝对路径；Git 中实际对应 `native/inputs/<key>_rgb.png`、`native/inputs/<key>_baseline.png` 和 `native/inference/<key>.png`。79 个完整 RGB 控制输入保留在服务器，Git 中包含其推理与逐像素复现记录。

本次没有新增正式指标。比较最终建图效果时继续使用仓库统一 v3 协议，不能把此目录中的局部分割数量、显示配色 IoU 或 24 个证据像素统计当作 v3 质量分数。
