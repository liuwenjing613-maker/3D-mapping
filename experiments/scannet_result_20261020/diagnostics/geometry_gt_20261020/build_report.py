from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
DELIVERY = WORK / "results/ScanNet_E1_auto-v3-swap_partial_20261020"


def read_ply(path):
    type_map = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    with path.open("rb") as f:
        props, n, vertex = [], None, False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element "):
                parts = line.split()
                vertex = parts[1] == "vertex"
                if vertex:
                    n = int(parts[2])
            elif line.startswith("property ") and vertex:
                _, t, name = line.split()
                props.append((name, type_map[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, np.dtype(props), count=n)


checks = []
summaries = {}
for scene in ["scene0011_00", "scene0608_00"]:
    folder = ROOT / scene
    geometry = json.loads((folder / "geometry.json").read_text())
    apertures = json.loads((folder / "window_apertures.json").read_text())
    sources = json.loads((folder / "pose_source_audit.json").read_text())
    pairs = json.loads((folder / "frame_pair_consistency.json").read_text())
    regions = json.loads((folder / "raw_depth_regions.json").read_text())
    with np.load(folder / "geometry_samples.npz") as z:
        gt = z["gt_xyz"]
        sampled = z["xyz"]
    for method in ["baseline", "E1", "swap_partial"]:
        path = DELIVERY / scene / "ply" / f"{scene}_{method}_instances.ply"
        a = read_ply(path)
        selected = a["surface_index"] % 10 == 0
        idx = a["surface_index"][selected] // 10
        xyz = np.column_stack([a[x][selected] for x in ["x", "y", "z"]])
        np.testing.assert_array_equal(xyz, sampled[idx])
        checks.append(dict(file=str(path.relative_to(WORK)), kind="prediction", checked_vertices=len(xyz),
                           coordinate_match_exact=True, full_vertices=len(a)))
    for suffix in ["", "_observed"]:
        path = DELIVERY / scene / "gt_ply" / f"{scene}_gt_instances{suffix}.ply"
        a = read_ply(path)
        xyz = np.column_stack([a[x] for x in ["x", "y", "z"]])
        np.testing.assert_array_equal(xyz, gt[a["reference_vertex_index"]].astype(np.float32))
        checks.append(dict(file=str(path.relative_to(WORK)), kind="GT", checked_vertices=len(xyz),
                           coordinate_match_exact=True, full_vertices=len(a)))
    totals = apertures["totals"]
    g = geometry["published_geometry_scope"]["swap_partial"]
    summaries[scene] = dict(
        whole_map_median_GT_distance_mm=geometry["distance_to_full_GT_triangle_mesh"]["mm_p10_p50_p90_p99"][1],
        rigid_adjustment_translation_mm=geometry["rigid_alignment_diagnostic"]["translation_mm"],
        rigid_adjustment_rotation_deg=geometry["rigid_alignment_diagnostic"]["angle_deg"],
        window_missing_fraction=totals["missing"] / totals["pixels"],
        window_accepted_behind_10cm_fraction=totals["accepted_behind_10cm"] / totals["accepted"],
        window_ghost_trace=apertures["ghost_trace"],
        published=g, published_floor_wall_fraction=(g["near_GT_floor"] + g["near_GT_wall"]) / g["published"],
        exact_source_frames=sources["frame_count"], adapter_depth_mismatch=sources["depth_mismatch_frames"],
        adapter_pose_error=sources["max_adapter_pose_error"], frame_pair_summary=pairs["summary"])
    with np.load(folder / "window_aperture_example.npz") as z:
        fid, roi, depth, plane = int(z["frame"]), z["roi"], z["depth"], z["window_plane_depth"]
    rgb = np.asarray(Image.open(folder / f"window_aperture_frame{fid}_rgb.jpg"))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    ax[0].imshow(rgb)
    ax[0].contour(roi, levels=[.5], colors=["#00e4e4"], linewidths=1)
    ax[0].set_title(f"Frame {fid}: diagnostic window aperture")
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#dcdcdc")
    im = ax[1].imshow(np.ma.masked_where(depth == 0, depth), cmap=cm, vmin=0, vmax=4)
    ax[1].contour(roi, levels=[.5], colors=["#ff5544"], linewidths=.8)
    ax[1].set_title("Raw depth (m); gray = no return")
    category = np.zeros(roi.shape, np.uint8)
    category[roi & (depth == 0)] = 1
    category[roi & (depth >= 4)] = 2
    category[roi & (depth > 0) & (depth < 4)] = 3
    category[roi & (depth > 0) & (depth < 4) & (depth > plane + .1)] = 4
    colors = np.array([[245, 245, 245], [135, 135, 135], [231, 167, 38], [45, 156, 120], [221, 70, 65]], np.uint8)
    ax[2].imshow(colors[category])
    ax[2].set_title("Gray: missing | Gold: >=4m\nGreen: valid | Red: >10cm behind aperture")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1], shrink=.8)
    fig.suptitle(scene + " | Diagnostic only; no masks or geometry modified", fontsize=13)
    fig.savefig(folder / "window_aperture_diagnostic.png", dpi=170)
    plt.close(fig)

summary = dict(scopes="Frozen-map diagnostic, not official instance scoring", scenes=summaries,
               ply_coordinate_checks=checks, maps_changed=False, selection_masks_changed=False,
               annotations_changed=False)
(ROOT / "diagnosis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
s = summaries["scene0011_00"]
t = summaries["scene0608_00"]
lines = [
"ScanNet 建图与 GT 的错位、玻璃/反光诊断（2026-10-20）",
"",
"结论：窗户确实存在严重深度缺测，和透明/反光表面的测量困难相符，但不能据此把截图中的全部差异归因于反光。已排除当前适配及 PLY 导出的整体坐标变换错误。实际差异包含：展示范围不同、原始深度与给定位姿的局部不一致、固定 TSDF 保留下来的局部表面差异。仅凭现有 RGB-D 不能把反光、透射、低回波和其他传感器因素严格拆分。",
"",
"一、不是 E1 或 partial 把点移动了",
"两个场景严格无修复、E1、partial 的完整 XYZ 逐点一致，并与 10 月 8 日修正后的固定 TSDF 完全一致；GT XYZ 与官方原始网格一致，两侧均未应用 axisAlignment。",
"238 + 339 个输入帧全部核对：深度 PNG 像素完全相同，适配位姿最大误差为 0；内参与已导出的 .sens 内参一致。此前 .sens 审计也确认抽查帧的时间戳、深度和位姿导出一致。",
"本次另对本地交付 PLY 核验：6 个预测 PLY 的每 10 个原始表面索引抽查坐标完全一致，4 个 GT PLY 全部顶点与原始 GT 一致，具体数量见 diagnosis_summary.json。",
f"scene0011_00 全部 TSDF 点到完整 GT 三角面的距离中位数为 {s['whole_map_median_GT_distance_mm']:.2f} mm；稳健整体配准只得到 {s['rigid_adjustment_translation_mm']:.2f} mm 平移、{s['rigid_adjustment_rotation_deg']:.3f}° 旋转。配准前后 >10 cm 的比例几乎不变，说明一个整体平移/旋转解决不了局部差异。scene0608_00 对应配准为 {t['rigid_adjustment_translation_mm']:.2f} mm、{t['rigid_adjustment_rotation_deg']:.3f}°。诊断变换没有应用到结果。",
"",
"二、截图左右不是相同的显示范围",
"交付 GT 实例 PLY 仅包含本次评价的目标物体；预测实例 PLY 包含所有已发布实例，也包括被赋实例 ID 的地面、墙面。",
f"scene0011_00 的 partial 共 {s['published']['published']:,} 个已发布点，其中 {s['published']['near_GT_floor']:,} 个距 GT 地板三角面不超过 3 cm，{s['published']['near_GT_wall']:,} 个距墙面不超过 3 cm，合计 {s['published_floor_wall_fraction']:.1%}。截图中新增的大块地面/墙面主要是这一范围差异，不能直接判为几何错位。",
"同坐标、同视角、同尺度对照：各场景 same_coordinates_comparison.png。灰色为完整 GT，绿色为距完整 GT 不超过 3 cm 的已发布点，红色为超过 10 cm 的点。GT 经清理和简化，红色仅表示差异，不自动等于错误几何。",
"",
"三、窗户确实深度不可靠，但反光不是已证实的唯一原因",
"直接只检查 GT 保留下来的窗户三角面会漏掉玻璃空洞，因此额外用官方窗口顶点拟合平面和窗口凸包；扣除前景遮挡，向内缩 2.5 cm，图像再侵蚀，检查整个窗口孔洞。它只定义诊断区域，没有参与建图或掩码选择。",
f"scene0011_00：窗口共 {106539:,} 个采样像素×帧，{74268:,} 个深度缺失（{s['window_missing_fraction']:.1%}），2,623 个 >=4 m 被现有规则排除；0–4 m 有效读数中 4,065 / 29,648（{s['window_accepted_behind_10cm_fraction']:.1%}）落在窗口参考平面后方超过 10 cm。",
f"scene0608_00：窗口缺失 {109500:,} / {124422:,}（{t['window_missing_fraction']:.1%}）。这些是沿采样像素和帧汇总的比例，不是独立玻璃表面积比例。",
"窗口后的回波可能来自真实窗外物体，也可能涉及透射、反射或测量噪声；本数据没有单独区分这些光学机制的证据。因此能够确认的是窗口深度缺测/穿透式读数，不能把所有后方点直接叫作反射伪影。",
"对 scene0011_00 距完整 GT >10 cm 的固定表面抽样点，17,756 个中 15,230 个能找到 2 cm 内的原始深度支持；449 个支持射线曾穿过诊断窗口，其中 347 个落在窗口平面后方超过 10 cm。窗口解释不了其余大多数差异点。参考网格未覆盖区域仍可能是真实场景，不能把这些差异全部视为噪声。",
"原图与区域核查：window_aperture_diagnostic.png、raw_depth_examples.png。",
"",
"四、局部观测不一致在建图之前就存在",
"不使用 GT，直接把原始深度按给定位姿投到另一帧：排除深度边缘、明显遮挡和法向差异后，仍存在厘米级残差。",
"scene0011_00 的 1190→1200 帧有 1,264 个符合筛选条件的对应点，点到平面残差 90 分位 25.65 mm，27.61% 超过当前正向投票的 15 mm 距离阈值；这一事实不依赖 GT 网格清理。",
"scene0011_00 汇总相邻帧残差中位数/90分位为 2.65/10.29 mm，重访帧为 3.49/13.47 mm；scene0608_00 相邻帧为 4.77/17.38 mm，重访帧为 4.55/15.60 mm。是有条件的诊断采样，不是全场景误差率。",
"逐帧与 GT 的诊断配准也显示局部残差：例如 scene0011_00 第 1300 帧，在初始距离 <8 cm 的平滑深度点上，独立保留点的距离 90 分位从 42.48 mm 降到 17.03 mm。但这只能说明刚性校正可解释部分残差，不能直接认定官方位姿有某个精确偏差；传感器深度误差、标定残差、GT 重建差异仍可能混合。",
"当前 TSDF 使用给定位姿直接融合，没有针对这里的局部观测残差再做配准。E1/partial 在固定几何上调整实例证据，不会修正这些 XYZ；改变第四步筛选阈值也不会修正几何。",
"",
"本轮边界与下一步",
"已确定的来源层级：原始观测/既有几何和展示范围，而非当前 E1/partial 或 PLY 导出造成的整体错位。“反光是主因”尚未得到证实；反光、透射、环境光、深度噪声和局部位姿残差的贡献尚不能唯一分离。",
"本轮没有修改算法、阈值、位姿、TSDF、预测、评价范围或人工标注。若后续进行修正，应先独立做局部几何一致性/可靠性对照，再决定是否更换固定几何；任何实现先按项目要求由用户确定。",
"",
"外部资料（仅解释可能机制，不代替当前场景证据）：",
"Structure 官方说明反光表面会带来跟踪丢失和伪影：https://support.structure.io/article/261-can-i-use-the-structure-sensor-to-capture-3d-scans-of-shiny-reflective-surfaces",
"ScanNet 官方说明采集传感器、重建及清理后的标注网格：https://www.scan-net.org/ScanNet/",
]
(ROOT / "错位与反光诊断.txt").write_text("\n".join(lines), encoding="utf-8")
manifest = {}
for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix in [".py", ".json", ".npz", ".png", ".jpg", ".txt"] and p.name != "diagnosis_manifest.json":
        manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / "diagnosis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(dict(status="PASS", checks=checks, summary=summaries), ensure_ascii=False))
