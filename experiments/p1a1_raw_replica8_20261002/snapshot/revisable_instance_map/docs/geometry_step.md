# 第 4 步：共享 TSDF 几何烟测

本步使用固定输入协议中的 room0 帧 0、5、…、45，从空图建立一份场景级 TSDF。几何积分只使用 RGB-D、内参和相机位姿；输入加载器虽读取 mask，几何模块不使用 mask，也不创建实例。

实现位于 `src/revisable_instance_map/geometry_tsdf.py`，运行入口为 `tools/smoke_tsdf.py`。参数：Open3D 0.19.0、CPU、1 cm 体素、每块 8³ 体素、初始容量 20,000 块、深度上限 10 m。相机的 camera-to-world 位姿在传给 Open3D 前取逆，成为 world-to-camera 外参。

结果位于 `/data/chenkejun/CVPR/revisable_instance_map/geometry_smoke_10/`，包含可重载的 `tsdf_grid.npz`、提取的 `surface.ply`、首帧光线投射深度和 `geometry_smoke.json`。

10 帧共激活 14,845 个体素块，提取 434,158 个表面点。首帧光线投射覆盖图像约 99.63%，与原深度在共同有效像素上的绝对误差中位数约 5.8 mm。表面点到参考网格的最近距离中位数约 5.5 mm；参考网格仅用于几何检查，未进入建图。保存的 TSDF 重新加载后，体素块数和表面点数保持一致。

当前峰值进程内存约 764 MB，TSDF 文件约 146 MB。20,000 块容量在 10 帧后已使用约 74%，不能直接假定足够容纳 400 帧。完整场景的容量、内存和分辨率需要下一步单独测试。本步尚无实例归属、关联、mask 修订或地图评估。
