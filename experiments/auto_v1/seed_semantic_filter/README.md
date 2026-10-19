# Auto-v1 最终种子整图语义筛选实验

最新用户要求：页面与下载清单只显示本轮“排除18／保留72”两类，达到固定0.7／0.9排除规则即排除，其余全部保留，不另设待复核组。`export_binary.py` 从已冻结结果生成 `binary_comparison.json` 和 `selection_binary.json`，未更改任何排除决定；以下25／47为此前诊断统计。

2026-10-18，按用户确认改用“整图语义分割＋原 Mask 内类别占比”。不改变 auto-v1 候选生成、最终 90 个 Mask 或编号；不做去重、不运行 E1。

使用 NVIDIA `nvidia/segformer-b5-finetuned-ade-640-640`，固定版本 `739f5d4692954e4a185eac280dec1ba5a7d52f1d`。官方权重 339,299,397 字节，SHA256 `a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df`。直连不可达时经 hf-mirror 下载，权重与官方 SHA256 一致；仅使用 `torch.load(weights_only=True)` 加载。文件下载耗时约 72 秒。

## 输入与固定规则

- 原始 r4 清单：90 个最终主种子、56 张 RGB；来源分别为原图 CropFormer 59、局部 CropFormer 10、SAM 21。
- 使用官方预处理将完整 RGB 缩放为 640×640，按官方均值方差归一化。FP32 推理，TF32 关闭。
- 先将 logits 双线性还原至原图 680×1200，再计算 softmax 和 argmax；与官方后处理的标签结果逐像素核对一致。`reduce_labels` 不用于预测编号。
- 从模型 `id2label` 找到 `wall/floor/ceiling/rug`，不猜类别索引，不新增文本评分。
- 高置信排除类像素 = argmax 属于四类且最大 softmax ≥ 0.7。其像素数除以原 Mask 面积 ≥ 0.9 时建议排除。高置信其他类占比 ≥ 0.9 时明确保留；其余待复核并保留。边界包含等号。
- 每帧只推理一次，同帧各来源 Mask 共享预测。保存原尺寸 uint8 标签和 FP32 置信度、输入哈希、模型版本、类别统计、全部决策。人工清单与 GT 均不进入推理。
- 0.6/0.7/0.8 置信度和 0.8/0.9/0.95 占比仅做事后敏感性展示，不改变固定 0.7/0.9 决策。

## 结果与边界

建议排除 18、明确保留 25、待复核保留 47，实际建议保留总计 72。56 帧模型推理合计 2.834 秒，含加载、预览导出和校验共 25.250 秒；测试使用 RTX 5880 Ada，16 项测试本地与服务器均通过。时间与 CLIP 运行的预览、缓存工作量不同，不当作严格速度基准。

与实际用于上一轮 E1 的人工 revision 67 对比：双方排除 16、程序排除但人工保留 2、人工排除但程序保留 46、双方保留 26。两个分歧为 G016（地毯，本轮新允许排除）与 G030（人工保留的长条板，模型当墙）。人工删除混有重复种子，不能把 46 项直接算语义漏检。

对上一轮已经逐图检查的 10 个明显墙／地／顶案例，CLIP 筛出 4，本轮筛出 10；这是案例对照，不是独立测试集召回率。原有 6 个地毯案例，CLIP 筛出 5，本轮筛出 3。G024/G044/G059 地毯碎片仍漏筛，G044/G059 大部分被判成椅子。

事后逐图复核 18 个排除建议：14 个与墙／地／顶／地毯范围一致，G030/G063/G065 有附着物误判或与人工保留目标冲突，G088 小条带无法可靠定类。G063/G065 是通风口却全判为 ceiling；虽然人工也删除，不能据此认定其语义判断正确。模型高置信错误不会被单纯提高占比门槛解决。详见 `visual_audit.json`，这些视觉判断并非用户补充的类别真值，也未用于改写自动决策。

## 路径与复现

代码：`/home/chenkejun/CVPR/experiments/seed_semantic_filter_20261018`。

模型：`/data/chenkejun/CVPR/models/segformer-b5-finetuned-ade-640-640/739f5d4692954e4a185eac280dec1ba5a7d52f1d`。

结果：`/data/chenkejun/CVPR/results/seed_semantic_filter_20261018`。

```text
python -B -m unittest -v test_rules test_compare
python -B run.py --manifest /data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4/seed_manifest.json --raw-config /data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json --out <新结果目录>
python -B compare.py --results <结果目录> --human-selection <已确认的 revision 67 清单> --clip-results <上一轮固定 classification.json>
```

`seed_manifest.json` 是实验建议清单，保留原始稀疏 `original_track_id`。不得将剩余种子重新从 1 编号后直接送旧 E1；本轮没有执行追踪或改图。后续如评估 E1，仍必须从服务器 main 当前默认入口使用 `object_observed_repair / revision 2`。

官方模型：https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640

官方后处理说明：https://huggingface.co/docs/transformers/model_doc/segformer
