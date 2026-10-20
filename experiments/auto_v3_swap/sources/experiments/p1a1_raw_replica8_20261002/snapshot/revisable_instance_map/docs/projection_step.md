# 第 3 步：读帧与 RGB-D 投影

`src/revisable_instance_map/frame_io.py` 按已冻结的 room0 输入协议读取 RGB、16 位深度、帧内 mask 和 `traj.txt`。原始深度除以 `6553.5` 得到米；只投影有限且大于零的深度像素。

对像素 `(u,v)` 和深度 `z`：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
p_world = T_camera_to_world * [x, y, z, 1]
```

投影结果保留原始帧号、像素坐标、RGB 和**帧内** mask ID，方便后续记录来源。mask 不控制几何点是否保留；本步没有跨帧身份、实例关联或融合。

服务器数据盘上的 `projection_smoke/projection_smoke.json` 记录帧 0、1000、1995 的每 8 像素抽样检查；`projection_smoke_full/projection_smoke.json` 记录帧 0 的全分辨率检查。PLY 仅为可视化用的抽样点云。抽样检查只读取参考网格的 `xyz` 计算最近距离，未读取 GT 实例或语义标签，也没有将参考网格传给建图模块。
