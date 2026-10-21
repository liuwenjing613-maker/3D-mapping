# partial＋E1＋swap 当前冻结基线

2026-10-21 按人工确认将现有 partial＋E1＋swap 提升为 main 的当前算法基线，版本 `partial-e1-swap-20261021`，算法 `frozen=true`。继承已验证的 `auto-v3-swap` 提交 `1103d8196a980874ef743f572d3b4c0748a4fdab`；本次只登记默认版本、锁定现有源码和结果，没有更改数值算法或重新生成地图。

## 固定内容

自动种子筛选沿用 swap 顺序：完整支持包含先于覆盖充分筛选，窗户/柜子仅豁免第五项 p/r 筛选。Replica 八场景共637个原始主种子、196个保留种子：room0=35、room1=22、room2=35、office0=19、office1=22、office2=23、office3=22、office4=18。room0 G030/M0466 的 CropFormer 来源修正，以及 room2 G017/G025 两个人工恢复例外继续保留，不能称为完全无人工干预。

修复顺序固定为 partial 直接撤换 → 原核心历史重关联 → 原 E1 自适应历史扩展 → 原最终后处理。partial 仅撤换其允许旧身份范围内的新 Mask 覆盖部分；同帧同点不同身份严格弃权。最终结果键沿用历史名称 `whole_to_partial`，此名称不表示当前采用 whole。默认验收的是该键的 **final** 地图，同时保留 native 结果供诊断。

原 E1 内核仍由 `instance_repair/configs/current_architecture.json` 注册；当前完整流程由 `instance_repair/configs/current_auto_pipeline.json` 注册并指向本目录的 `config.json`。上游 `experiments/auto_v3_swap/config.json` 中的 whole 默认值是原版本历史记录，不再决定 main 当前基线。onebyone、人工76种子实验和新的保护规则不在本次冻结范围。

## 来源与复现

`config.json` 固定本版本选择；`result_lock.json` 列出八场景真实结果目录、native/final 地图与评价锁；`server_verification.json` 保存服务器实文件核验结果；`evidence/<scene>` 保存当时的预测冻结清单、partial 策略、评价来源与指标；`release_manifest.json` 固定本目录交付文件及注册文件字节。上游源码、模型及637张源 Mask 的验证仍使用未改动的 auto-v3-swap 源码锁。

代码、配置及文档在 `/home/chenkejun/CVPR`，大数据、跟踪结果和 PLY 保留在 `/data/chenkejun/CVPR`，通过来源路径和 SHA-256 定位，不复制进 Git。八场景统一入口为 `/data/chenkejun/CVPR/results/replica8_swap34_20261020/<scene>/partial`；部分入口是指向已核验旧结果的符号链接。以 `result_lock.json` 的真实源路径为准，尤其 office1、office3、office4 必须使用 swap 后重跑结果。

```bash
# 在仓库根目录验证当前注册、源码快照、八场景选择、结果证据和评价锁。
python -B experiments/partial_e1_swap_baseline_20261021/verify.py
# 在原服务器追加核对模型、源 Mask 和所有锁定外部结果文件。
python -B experiments/partial_e1_swap_baseline_20261021/verify.py --server-sources
```

完整模型/地图重算需要原服务器数据和环境。重跑必须创建独立代码与结果目录，不能直接调用历史 launcher 覆盖已冻结结果；只做路径适配时记录差异，数值算法变化另起版本。上游按场景执行入口见 [auto-v3-swap 复现说明](../auto_v3_swap/README.md)。

## 当前评价结果

沿用 main 的 `object_observed_repair / revision 2`、350个固定可评价 GT；评价协议 `frozen=false`。评分源码 `cd260569e89de1e9525f82c45ee10842b80616b0`、审核归档 `d035f68770cefab27c9751e97062c72faa07367c` 均不改变。评价从 `/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009` 的默认入口运行。

八场景等权宏平均，数值来自已完成的 R2 评价，本次不重新评分：

| 结果 | F1 | PQ | AP |
|---|---:|---:|---:|
| partial＋E1＋swap，native | 0.620023 | 0.484685 | 0.187685 |
| **partial＋E1＋swap，final（当前基线）** | **0.624777** | **0.496920** | **0.199145** |
| 无修复，final（对照） | 0.627947 | 0.490819 | 0.187719 |

本次冻结用于后续稳定比较，不代表已解决所有损伤，也不代表全部指标优于无修复。逐场景指标与实际预测通过同一来源锁关联，禁止用 PLY 颜色代替原生实例 ID 评分。
