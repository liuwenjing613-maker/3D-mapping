# CVPR 当前开发评价基准

- 后续统一使用 `unified_eval/configs/replica_ca_v3.object_observed_repair.audit_r2.json`，`object_observed_repair / revision 2`。当前选择与来源锁见 `unified_eval/configs/current_protocol.json`。
- 评分源码固定为 `cd260569e89de1e9525f82c45ee10842b80616b0`；审核归档固定为 `d035f68770cefab27c9751e97062c72faa07367c`。更新 main 的入口和说明不改变这两个来源提交的含义。
- GT 为八场景共 350 个固定可评价对象。6 个极小 GT 继续待审，168 个 UNKNOWN 保持 UNKNOWN。不重新生成或扩大当前评价范围。
- 用于固定 TSDF 上的实例地图质量与成对标签修复。成对修复必须共用相同几何、GT 范围和对应索引；最终 400 帧范围不能用于在线前缀评价。
- 当前入口：`python -m unified_eval.cli current-protocol`、`adapt-surface`、`eval-scene`、`eval-batch`、`eval-repair-pair`。默认 revision 2；旧协议复现须显式使用 `--historical-protocol`，结果单独标记，不能混入当前比较。
- 配置仍为开发基准、`frozen=false`。用户采用当前开发基准不等于批准论文正式冻结，不擅自修改参数、评分源码或这一标记。
- 服务器代码、配置、文档在 `/home/chenkejun/CVPR`；数据和结果在 `/data/chenkejun/CVPR`。新的评价使用独立输出目录，不覆盖已锁定的开发和审核归档。


## 场景最终发布策略（2026-11-01）

后续无修复、main、auto-v3-swap、online-v1/online-v2 全部采用 `scene-final-B-20261101`：整个场景及全部最终处理/末次修复完成后，统一执行一次 B，再以 B 产物评价和导出实例PLY。新实验默认使用 [统一整场景入口](docs/SCENE_FINAL_B.md)；不得在中间帧、检查点或修复过程中执行 B，不向在线状态回写补全标签，不启用 C。历史归档/结果锁保留 pre-B，当前种子边界策略及 E1/R2 规则不变。
