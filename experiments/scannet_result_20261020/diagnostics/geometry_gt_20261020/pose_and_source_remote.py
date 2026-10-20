"""Diagnose per-frame residuals; transforms are NEVER applied to map inputs."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from pathlib import Path
import json
import cv2
import numpy as np
import open3d as o3d

C = Path("/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020")
R = Path("/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020")
OUT = Path("/data/chenkejun/CVPR/results/scannet_geometry_gt_diagnosis_20261020")


def stat(x):
    return dict(n=len(x), mm_p50_p90_p99=(1000 * np.quantile(x, [.5, .9, .99])).tolist()) if len(x) else dict(n=0)


for scene in ["scene0011_00", "scene0608_00"]:
    out = OUT / scene
    config = json.loads((C / "configs" / (scene + ".json")).read_text())
    cam = config["camera"]
    raw = Path("/data/chenkejun/datasets/ScanNet/scans") / scene
    processed = Path("/data/chenkejun/CVPR/datasets/MaskClustering/scannet/processed") / scene
    mesh = o3d.io.read_triangle_mesh(str(raw / (scene + "_vh_clean_2.ply")))
    mesh.compute_vertex_normals()
    caster = o3d.t.geometry.RaycastingScene(nthreads=4)
    caster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    q = o3d.geometry.PointCloud()
    q.points, q.normals = mesh.vertices, mesh.vertex_normals
    q = q.voxel_down_sample(.015)
    poses = np.loadtxt(R / "inputs" / scene / "traj.txt").reshape(-1, 4, 4)
    mismatch_depth, max_pose_err = [], 0.
    for fid in config["frame_ids"]:
        a = cv2.imread(str(processed / "depth" / (str(fid) + ".png")), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(R / "inputs" / scene / "results" / f"depth{fid:06d}.png"), cv2.IMREAD_UNCHANGED)
        if not np.array_equal(a, b):
            mismatch_depth.append(fid)
        pp = np.loadtxt(processed / "pose" / f"{fid}.txt")
        max_pose_err = max(max_pose_err, float(np.max(np.abs(pp - poses[fid]))))
    records = []
    fids = sorted(set(config["frame_ids"][::5] + ([1190, 1200, 1250, 2010, 2370] if scene == "scene0011_00" else [1420, 1690, 3380])))
    for fid in fids:
        d = cv2.imread(str(processed / "depth" / (str(fid) + ".png")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        lo = cv2.erode(d, np.ones((5, 5), np.uint8))
        hi = cv2.dilate(d, np.ones((5, 5), np.uint8))
        v, u = np.mgrid[0:480:4, 0:640:4]
        ds = d[::4, ::4]
        valid = (ds > .3) & (ds < 3.5) & (lo[::4, ::4] > 0) & ((hi - lo)[::4, ::4] < .05)
        dr = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], axis=-1)
        pc = dr[valid] * ds[valid, None]
        pose = poses[fid]
        xyz = pc @ pose[:3, :3].T + pose[:3, 3]
        distance = caster.compute_distance(o3d.core.Tensor(xyz.astype(np.float32)), nthreads=4).numpy()
        eligible = distance < .08
        pts = xyz[eligible]
        if len(pts) < 500:
            continue
        # Deterministic spatial checkerboard holdout, no GT-derived online changes.
        yy, xx = v[valid][eligible], u[valid][eligible]
        train = ((xx // 16 + yy // 16) % 2) == 0
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts[train])
        reg = o3d.pipelines.registration.registration_icp(
            p, q, .08, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=.03)),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        T = reg.transformation
        moved = pts[~train] @ T[:3, :3].T + T[:3, 3]
        aft = caster.compute_distance(o3d.core.Tensor(moved.astype(np.float32)), nthreads=4).numpy()
        holdout = distance[eligible][~train]
        center = pose[:3, 3]
        rec = dict(frame=fid, fit_points=int(train.sum()), holdout_points=int((~train).sum()),
                   before=stat(holdout), after=stat(aft),
                   angle_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
                   camera_center_displacement_mm=float(np.linalg.norm(center @ T[:3, :3].T + T[:3, 3] - center) * 1000),
                   mean_observed_point_displacement_mm=float(np.mean(np.linalg.norm(moved - pts[~train], axis=1)) * 1000),
                   transform=T.tolist())
        records.append(rec)
    result = dict(scene=scene, frame_count=len(config["frame_ids"]), depth_mismatch_frames=mismatch_depth,
                  max_adapter_pose_error=max_pose_err, frame_level_alignment=records,
                  scope="diagnostic ICP on smooth valid depth, initial mesh distance<8cm; alternating image blocks for fit/held-out assessment; no transforms applied",
                  limitations="Rigid ICP improvement demonstrates pose/depth-to-reference residuals; it alone cannot distinguish physical pose error, depth bias, calibration error, or GT reconstruction changes.")
    (out / "pose_source_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    high = sorted(records, key=lambda x: -x["mean_observed_point_displacement_mm"])[:5]
    print(scene, "SOURCE", len(config["frame_ids"]), mismatch_depth, max_pose_err, flush=True)
    print(scene, "FRAME_ALIGNMENT", json.dumps([{k: r[k] for k in ["frame", "before", "after", "angle_deg", "camera_center_displacement_mm", "mean_observed_point_displacement_mm"]} for r in high]), flush=True)
