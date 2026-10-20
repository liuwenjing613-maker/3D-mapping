# room2 前30候选中已导出标注的联合修复

本轮仅执行 `room2_前30_修复候选选择_20261008.json` 内实际存在的18条选择：14组可用种子、25个mask，4处不可靠。12处没有标注，不自动补选、不按失败处理。原标注SHA256为 `6b3f32f5dc1c6ac96c940572efb12aad9d2ecf208288cf685823c172f50f6a69`。

代码放在 `/home/chenkejun/CVPR/experiments/room2_top30_repair_20261008/batch_6b3f32f5dc1c`；结果、日志放在 `/data/chenkejun/CVPR/revisable_instance_map/room2_top30_repair_20261008/batch_6b3f32f5dc1c`。基线仍是原始P1-A1 room2，未叠加之前5例修复。

## 保留的做法

沿用room0已经核验的联合换票代码。原始CropFormer选择在裁剪内逐像素核对后恢复对应整帧原mask；局部CropFormer和SAM2.1保留人工所选的原始裁剪mask，不推测补齐。人工允许触及裁剪边缘的选择保留，图像原本的边缘与人工裁剪边缘分别核对。

每组对2000个原始帧做独立种子状态的前后传播；只处理原有400个建图帧。数字质量门槛、投票单位/去重、原生确认条件、后处理和统一v3参数保持。共享旧家族的所有已选目标可靠时才可能整张撤票，否则只替换可靠前景下对应旧票。未选观察与范围外标签保留；共同表面的其他区域可能显示共享修复变化。

C0002 mask7与C0029 mask4是同一把椅子，C0002 mask8与C0029 mask10是另一把椅子。RGB种子和双向可见表面互投支持身份24→4、25→5，共23个身份；在换票和GT评估前冻结。所有25个原mask和28,000张实际传播结果仍保留。其他相同旧家族、不同部件或不确定选择不自动合并；GT不用于身份重组。

C0015 mask23只有14个有效种子表面点，低于原有20个可见点质量条件。仍完成追踪，实际是否换票以逐帧检查为准，没有因此放宽条件或隐藏失败。

## 必须通过的核验

逐像素来源、种子/配置/模型与原始RGB哈希；所有28,000张传播图和前后方向；13项已有联合操作核验；400帧完整重算等于增量票据；准确回滚；范围外证据、投票和最终标签不变；几何/RGB相同；预测在GT评估前冻结；PLY逐字段回读相等。

统一v3评估沿用既有代码、flags、协议与GT版本。当前协议是pending/debug，数值用于本轮离线对比，不自动作为正式主表。AP50/F1/PQ、固定目标最佳IoU、结构计数、灰点和非目标损伤全部报告；实例在视觉上分开，不自动认定整体修复成功。

## 执行

prepare_local.py → adapt_batch.py → prepare_server.py → tracking_queue.py → cross_seed_diagnostics.py → alias_review.py → run_batch.py → evaluate_batch.py → export_review.py → build_review.py → verify_review.py / final_audit.sh。

预测、原始数据和旧结果分别保留。本轮没有改几何、训练模型、搜索GT标签、调整评估参数或覆盖旧地图。实际指标、每个mask用帧情况、残留旧票、原始/仅种子/完整追踪三种真实PLY在独立结果页中呈现。
