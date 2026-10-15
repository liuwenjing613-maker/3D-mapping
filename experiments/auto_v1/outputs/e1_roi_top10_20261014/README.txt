room2 / E1 修复前三状态前十 / 2026-10-14

网页：http://127.0.0.1:8785/room2_E1_prerepair_CUT_top10_20261014/index.html
本地页面：results/固定案例_三模型对比_20261006/room2_E1_prerepair_CUT_top10_20261014/index.html
如本地服务未运行，执行 outputs/strict_vote_annotation_20261010/start_annotation.ps1。

输入：P1-A1-strict-vote 未人工修复地图。E1 main 架构保持不变。
服务器代码：/home/chenkejun/CVPR/experiments/e1_roi_top10_20261014
服务器数据：/data/chenkejun/CVPR/results/e1_roi_top10_20261014
基线：/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain/final/instance_surface.npz
基线 SHA256：3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3

按 C=3、U=0、T=1 且最终未归属的点形成现有完整一环连接 ROI。三类分别按绝对点数前10，相同点数按原连通分量编号排序。30个入榜名额对应19个不同ROI；跨榜重复仅展示一次，保留各榜排名、点数及每榜3个历史视角。共90个视角条目（可能包括同一物理帧）。每个裁剪并排展示原图CropFormer缓存、局部CropFormer、SAM2.1；412个裁剪，另有1个整帧复算对照。

ranking_audit.json 独立重建全部连通分量，校验三类前十、完整点索引和源文件哈希。两模型audit通过，CropFormer整帧对照与冻结缓存像素一致率1.0。同像素出现多个状态时仅显示当前入榜状态优先，原始三通道状态投影保留。projection_annotation_audit.json 验证90个视角全部目标状态像素均可显示，且各Mask直方图和锚点一致。

网页选中的Mask保存为PASS；都不可靠保存为FAIL。PASS仅指被选Mask，不是整张分割图。标注按19个唯一ROI保存，跨类别同步。新浏览器存储键与历史标注隔离。?qa=1用于隔离的功能测试记录。用户应导出 room2_E1修复前三状态前十_质量判定_20261014.json。

本轮没有运行追踪、修改地图或新增评分，没有使用GT选择ROI或Mask。后续评分仍须从main默认R2入口运行。

代码过程：assemble.py复用旧流程生成新实验代码；服务器pipeline.py运行audit_inputs、prepare_inputs、两模型推理、render_annotation（含normalize_manifest）；finalize_projections.py独立验证显示投影并封装；本地unpack_verify.py验证下载哈希；build_page.py生成页面；verify_delivery.py验证全部资源与元数据。
