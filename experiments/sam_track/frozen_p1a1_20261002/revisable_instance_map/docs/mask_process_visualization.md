# CropFormer、深度几何与 MaskFusion 可视化

可复现入口：`tools/visualize_mask_process.py`。图片只在服务器 `/data/chenkejun/CVPR/revisable_instance_map/mask_visualizations/`；未把 RGB-D、mask、图片或权重上传到 GitHub。输入是已有的真实 Replica room0 缓存，不重新分割、不使用 GT。

```bash
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python \
  tools/visualize_mask_process.py --frame-id 1705 --parent-id 5 --segment-id 46 \
  --output-dir /data/chenkejun/CVPR/revisable_instance_map/mask_visualizations
```

图像说明：

- `frame001705_overview.jpg`：同一 RGB 画面依次叠加原始 CropFormer mask、深度几何区域 mask、官方阈值的 OVI MaskFusion 结果；黄色轮廓始终是同一原始 CropFormer 提案 5。每栏的伪彩色仅区分该栏 ID，跨栏相同颜色不表示同一对象。
- `frame001705_focus_parent5.jpg`：放大沙发。一个原始提案与 10 个最终正细化片段相交，原始沙发像素中另有 3,621 个未进入正细化片段。片段数不是物体数。
- `frame001705_regrouped_parent5.jpg`：从原始 mask 到 MaskFusion，再到当前基础架构采用的逐像素真实来源归组。左、右两栏使用相同原始 ID 配色；右栏橙色是仍留在来源账本、当前未入图的原始残留像素。
- `frame001705_cross_source_segment46.jpg`：紫红轮廓是**同一个**融合片段 46。MaskFusion 以多数像素将它标为原始来源 12，但其中只有 233,943 像素确属来源 12，188,641 像素来自其他原始提案，1,005 像素来自原始背景。右栏青色/红色/橙色分别表示三者。这是不能把片段 `cropformer_id` 当作整片真实来源的直观例子。
- `frame001705_source_evidence_parent5.jpg`：沙发来源证据。青色为进入细化正区域的原始像素，橙色为残留。黄色线始终是原始提案边界。

本帧原始 mask 有 35 个正提案、深度几何 mask 有 43 个正区域、MaskFusion 有 48 个正片段。所有统计和四类输入 SHA-256 保存在 `frame001705_report.json`；渲染只改显示颜色，不改 mask 值。
