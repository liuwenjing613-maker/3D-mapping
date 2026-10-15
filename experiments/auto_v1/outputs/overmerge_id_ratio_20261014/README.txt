room2 / E1 修复前 / 初步过合并排除 / 2026-10-14

网页：http://127.0.0.1:8785/room2_E1_overmerge_audit_20261014/index.html
服务未运行时：powershell -ExecutionPolicy Bypass -File outputs/strict_vote_annotation_20261010/start_annotation.ps1

规则来自关联对话最新的无 VLM 初步方案：第二大 CONFIRMED 持久 ID 点数 n2 >= 10，且 n2 / N >= 0.20，两个条件同时成立才拟剔除。实现使用整数 5*n2 >= N，避免浮点阈值歧义。N 是 Mask 内全部有效可见的 TSDF 顶点数，U/T/CONFLICT 不作为其他物体，但计入分母。N=0 保留为无法判断。没有增加其他筛选门槛。

范围：原批次三状态各前十，30 个名额、19 个唯一 ROI、90 个视角条目、68 个物理帧、412 个裁剪。18,303 处 Mask 展示，按同帧同模型完整二值形状去重后 12,357 个候选。不同帧、不同模型分别计数，不能把该数解释为不同物体数。全帧对照推理任务不计入候选批次。

结果：拟剔除 194，保留 12,163（其中 46 个无有效支持）。原图 CropFormer 1,081 个去重候选中拟剔除 14；局部 CropFormer 4,117 个中 115；SAM2.1 7,159 个中 65。这是启发式触发数量，不是人工确认的过合并数量，不能据此比较模型真实质量。

输入语义：原图 CropFormer 检查完整原图 Mask，只纳入原裁剪中出现过的标签；局部 CF/SAM 检查既有可选标签区域的完整范围。SAM 使用既有“小区域优先”标签化结果，不对被覆盖的原始重叠 proposals 做本次筛选。局部边缘截断只提醒，不参与剔除。地图 native state=2 的 top1_instance_id 与 final instance_id 对所有 1,058,591 个确认顶点逐点一致。未把后处理填充标签当成 CONFIRMED。

几何：1,229,303 个固定 TSDF 顶点，复用 strict_vote_annotation 的 project_visible，RGB-D 三维回投误差 <=0.02 m。每个可见顶点只计一次，投影同一像素的不同顶点仍分别计数。图中点颜色用于定位，计数不来自彩色像素。

验证：服务器独立按 Mask 直接查询重算 1,177 项；本地从下载后的逐顶点投影和网页标签 PNG 重算全部候选，12,369 次来源形状检查、81,961,460 次点归属计数全部一致。所有 18,303 处来源可追溯，源地图、证据和分割结果哈希未变。见 audit.json、local_verification.json、browser_qa.json。

界面：默认查误删；查漏筛包括全部保留项，可筛选单 ID、无确认 ID、无 3D 支持等情况。候选详情同坐标展示 RGB、完整 Mask、第一/第二持久 ID 支持点。三模型总览可切换全部、筛选后、仅拟剔除；点击 Mask 回到详情。人工判断自动保存，导出带证据的 JSON。单物体被剔除记误删，多物体被保留记漏筛；不清楚可暂记无法判断。人工结果只统计已判定部分。

正式存储键：cvpr-room2-overmerge-id20-n10-20261014-v1；?qa=1 使用独立 -qa 键。测试记录不是人工真实判断。旧页面和旧标注不修改。

服务器代码：/home/chenkejun/CVPR/experiments/overmerge_id_ratio_20261014
服务器数据：/data/chenkejun/CVPR/results/overmerge_id_ratio_20261014
筛选清单：proposed_reject.json、retained.json。原始 Mask 不删除；本轮不执行追踪、修复、模型推理或评测，不使用 GT/VLM。
后续评价仍需 main 当前 object_observed_repair / revision 2 协议；本轮没有产生新的质量评分。

最主要的两类局限：一个真实物体有两个旧 ID 会误删；多个物体共享同一旧 ID 会漏筛。网页同时开放拟剔除和保留项供检查，不能在人工核验前断言筛得全面准确。

2026-10-14 界面调整：保留项检查改回原 ROI 三模型页面。
新页面：http://127.0.0.1:8785/room2_E1_retained_ROI_20261014/index.html
194 个去重拟剔除候选在原页面对应 298 处展示，已从 282 组模型裁剪中去除，涉及 212 个裁剪。所有 19 个 ROI、90 个视角条目、412 个裁剪保持原顺序与原交互。保留标签不重编号、不修改像素与颜色；拟剔除标签从 ID 图、着色、边界、投影关联、编号图例及可选目标中一起排除。静态对照图同步更新。
筛选页的保留项入口已跳转到此 ROI 页面。原始页面和原始标注保留；新记录使用 cvpr-room2-e1-retained-roi-20261014-v1。导出 JSON 含筛选来源，label_asset 对应原服务器标签图与原 ID，review_label_asset 对应筛选后审阅图。
服务器筛选后文件归档：/data/chenkejun/CVPR/results/overmerge_id_ratio_20261014/retained_ROI_review；网页依赖既有原始页面的共享 RGB 等资源。验证见 retained_verification.json、retained_browser_qa.json。

2026-10-14 显示开关：ROI 页面新增“显示拟剔除 Mask（对照检查）”总开关，各模型栏另有独立开关。默认隐藏；显示时使用原色填充、红色边界和“拟剔除 ID”标签。显示仅用于比较，不加入可选合格目标，保留的标签图、已选目标、正式存储键与既有记录不变。原始拟剔除标签图按需加载。重建页面后运行 add_rejected_toggle.py 恢复这部分界面；浏览器验证见 rejected_toggle_qa.json。

