"""Post-hoc diagnostic only: original maps, masks, annotations remain read-only.

GT is used solely to measure/reference already frozen geometry. Diagnostic ICP
transforms are not applied to any saved map or used in an evaluation.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import time
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

C = Path("/home/chenkejun/CVPR/experiments/scannet_e1_swap_partial_20261020")
R = Path("/data/chenkejun/CVPR/results/scannet_e1_swap_partial_20261020")
OUT = Path("/data/chenkejun/CVPR/results/scannet_geometry_gt_diagnosis_20261020")
OUT.mkdir(parents=True, exist_ok=True)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stats(x):
    x = np.asarray(x)
    if not len(x):
        return {"n": 0}
    return dict(n=len(x), mm_p10_p50_p90_p99=(np.quantile(x, [.1, .5, .9, .99]) * 1000).tolist(),
                abs_over_15mm=float(np.mean(np.abs(x) > .015)),
                abs_over_50mm=float(np.mean(np.abs(x) > .05)),
                abs_over_100mm=float(np.mean(np.abs(x) > .1)))


def cloud(x):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(x)
    return p


def distance(caster, x):
    return caster.compute_distance(o3d.core.Tensor(x.astype(np.float32)), nthreads=4).numpy()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scene_audit(scene):
    started = time.time()
    out = OUT / scene
    out.mkdir(exist_ok=True)
    config = json.loads((C / "configs" / (scene + ".json")).read_text())
    raw = Path("/data/chenkejun/datasets/ScanNet/scans") / scene
    processed = Path("/data/chenkejun/CVPR/datasets/MaskClustering/scannet/processed") / scene
    mesh = o3d.io.read_triangle_mesh(str(raw / (scene + "_vh_clean_2.ply")))
    ref = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    caster = o3d.t.geometry.RaycastingScene(nthreads=4)
    caster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    with np.load(Path("/data/chenkejun/CVPR/evaluation_results/scannet_v3_r2_reeval_20261009") / scene / "ground_truth/gt.npz") as z:
        gt_ref = z["xyz_ref"]
        gt_ids = z["raw_instance_id"].copy()
        target = z["instance_id"] > 0
        region = z["evaluation_region"].copy()
        meta = json.loads(str(z["metadata_json"].item()))
    np.testing.assert_array_equal(gt_ref, ref.astype(gt_ref.dtype))
    names = {x["objectId"] + 1: x["label"] for x in json.loads((raw / (scene + ".aggregation.json")).read_text())["segGroups"]}
    names[-1] = "unannotated"
    tri_ids = gt_ids[tri[:, 0]].copy()
    tri_ids[np.any(gt_ids[tri] != tri_ids[:, None], axis=1)] = -1
    name_list = sorted(set(names.values()))
    name_to_num = {n: i for i, n in enumerate(name_list)}
    tri_classes = np.array([name_to_num[names.get(int(x), "unannotated")] for x in tri_ids])
    window_c = name_to_num.get("window", -100)
    source_paths = {
        "baseline": R / "baseline/ablation" / scene / "abstain/final/instance_surface.npz",
        "E1": R / scene / "e1_repair/e1/E1_expanded_history" / scene / "final/instance_surface.npz",
        "swap_partial": R / scene / "partial/whole_to_partial" / scene / "final/instance_surface.npz",
    }
    xyz = None
    labels = {}
    for method, p in source_paths.items():
        with np.load(p) as z:
            if xyz is None:
                xyz = z["xyz_m"].copy()
            else:
                np.testing.assert_array_equal(xyz, z["xyz_m"])
            labels[method] = z["instance_id"].copy()
    geometry_path = R / "inputs" / scene / "geometry/surface.ply"
    historical_xyz = np.asarray(o3d.io.read_point_cloud(str(geometry_path)).points)
    np.testing.assert_array_equal(xyz, historical_xyz.astype(xyz.dtype))
    full_dist = distance(caster, xyz)
    nearest = caster.compute_closest_points(o3d.core.Tensor(xyz.astype(np.float32)), nthreads=4)
    nearest_tri = nearest["primitive_ids"].numpy().astype(np.int64)
    nearest_class = tri_classes[nearest_tri]
    nearest_obj = tri_ids[nearest_tri]
    geometry = dict(points=len(xyz), distance_to_full_GT_triangle_mesh=stats(full_dist),
                    all_three_methods_xyz_exactly_equal=True,
                    historical_engineered_geometry_exactly_equal=True,
                    GT_xyz_exactly_raw_mesh=True,
                    GT_coordinate_transform=meta["coordinate_transform"],
                    source_sha256={k: sha(v) for k, v in source_paths.items()},
                    geometry_source=str(geometry_path.resolve()),
                    reference_mesh_sha256=sha(raw / (scene + "_vh_clean_2.ply")),
                    per_nearest_class={n: stats(full_dist[nearest_class == i]) for i, n in enumerate(name_list)})
    foreground = {}
    for method, lab in labels.items():
        positive = lab > 0
        floor = (nearest_class == name_to_num.get("floor", -2)) & (full_dist <= .03)
        wall = (nearest_class == name_to_num.get("wall", -2)) & (full_dist <= .03)
        foreground[method] = dict(published=int(positive.sum()), near_GT_floor=int((positive & floor).sum()),
                                  near_GT_wall=int((positive & wall).sum()),
                                  over_10cm_from_full_GT=int((positive & (full_dist > .1)).sum()))
    geometry["published_geometry_scope"] = foreground
    # A single rigid correction diagnostic fitted on close surfaces only.
    p = cloud(xyz).voxel_down_sample(.04)
    px = np.asarray(p.points)
    pd = distance(caster, px)
    p = p.select_by_index(np.flatnonzero(pd < .05))
    mesh.compute_vertex_normals()
    q = cloud(ref)
    q.normals = mesh.vertex_normals
    q = q.voxel_down_sample(.025)
    reg = o3d.pipelines.registration.registration_icp(
        p, q, .05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=.015)),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T = reg.transformation
    idx = np.arange(0, len(xyz), 10)
    moved = xyz[idx] @ T[:3, :3].T + T[:3, 3]
    geometry["rigid_alignment_diagnostic"] = dict(
        matrix=T.tolist(), angle_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
        translation_mm=float(np.linalg.norm(T[:3, 3]) * 1000),
        displacement_at_map_points_mm=stats(np.linalg.norm(moved - xyz[idx], axis=1)),
        before=stats(full_dist[idx]), after=stats(distance(caster, moved)),
        applied_to_maps=False, fit_scope="downsampled points within 5cm of full GT, robust Tukey 1.5cm")
    save(out / "geometry.json", geometry)
    np.savez_compressed(out / "geometry_samples.npz", xyz=xyz[idx], distance_m=full_dist[idx],
                        near_class=nearest_class[idx], class_names=np.array(name_list),
                        baseline=labels["baseline"][idx], E1=labels["E1"][idx],
                        swap_partial=labels["swap_partial"][idx],
                        gt_xyz=ref, gt_target=target, gt_region=region, gt_raw_ids=gt_ids,
                        gt_rgb=np.asarray(mesh.vertex_colors))
    print(scene, "GEOMETRY", json.dumps({k: geometry[k] for k in ["distance_to_full_GT_triangle_mesh", "rigid_alignment_diagnostic", "published_geometry_scope"]}), flush=True)

    cam = config["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    originalK = np.loadtxt(processed / "intrinsic/intrinsic_depth.txt")[:3, :3]
    assert np.max(np.abs(K - originalK)) < 1e-5
    poses = np.loadtxt(R / "inputs" / scene / "traj.txt").reshape(-1, 4, 4)
    v, u = np.mgrid[0:480:4, 0:640:4]
    camera_rays = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], axis=-1)
    # Each row is pooled pixel/frame evidence, not independent physical samples.
    accumulated = defaultdict(lambda: defaultdict(list))
    per_frame = []
    ghosts_idx = np.flatnonzero(full_dist[idx] > .1)
    ghost = xyz[idx[ghosts_idx]]
    gs = np.zeros(len(ghost), np.uint16)
    gw = np.zeros(len(ghost), np.uint16)
    gbehind = np.zeros(len(ghost), np.uint16)
    gother = np.zeros(len(ghost), np.uint16)
    galign = np.zeros(len(ghost), np.uint16)
    examples = []
    for seq, fid in enumerate(config["frame_ids"]):
        d_full = cv2.imread(str(processed / "depth" / (str(fid) + ".png")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        d = d_full[::4, ::4]
        pose = poses[fid]
        directions = camera_rays @ pose[:3, :3].T
        origins = np.broadcast_to(pose[:3, 3], directions.shape)
        hit = caster.cast_rays(o3d.core.Tensor(np.concatenate([origins, directions], axis=-1).astype(np.float32)), nthreads=4)
        depth_ref = hit["t_hit"].numpy()
        hit_ids = hit["primitive_ids"].numpy().astype(np.int64)
        hit_ok = np.isfinite(depth_ref)
        cls = np.full(d.shape, -1, dtype=np.int32)
        cls[hit_ok] = tri_classes[hit_ids[hit_ok]]
        roi = hit_ok & (depth_ref > .2) & (depth_ref < 4)
        # Compare away from GT depth jumps and class boundaries.
        rd = np.where(hit_ok, depth_ref, 0).astype(np.float32)
        smooth = cv2.dilate(rd, np.ones((3, 3), np.uint8)) - cv2.erode(rd, np.ones((3, 3), np.uint8)) < .08
        for c in np.unique(cls[roi]):
            name = name_list[c]
            mask = roi & (cls == c)
            interior = mask & smooth & (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            valid = interior & (d > 0)
            accepted = valid & (d < 4)
            rec = dict(frame=fid, region=name, roi=int(mask.sum()), interior=int(interior.sum()),
                       missing=int((interior & (d <= 0)).sum()), over4=int((interior & (d >= 4)).sum()),
                       accepted=int(accepted.sum()), behind10=int((accepted & (d - depth_ref > .1)).sum()),
                       front10=int((accepted & (d - depth_ref < -.1)).sum()))
            if rec["interior"]:
                per_frame.append(rec)
                acc = accumulated[name]
                acc["counts"].append([rec[x] for x in ["interior", "missing", "over4", "accepted", "behind10", "front10"]])
                acc["accepted_residual"].append((d - depth_ref)[accepted])
                acc["raw_valid_residual"].append((d - depth_ref)[valid])
                if name == "window" and rec["interior"] >= 100:
                    examples.append((rec["behind10"] + rec["missing"] + rec["over4"], fid, rec))
        # Trace fixed-map points >10cm from GT to their supporting measured rays.
        camera_xyz = (ghost - pose[:3, 3]) @ pose[:3, :3]
        z = camera_xyz[:, 2]
        uv = camera_xyz @ K.T
        uv = np.rint(uv[:, :2] / np.maximum(z[:, None], 1e-9)).astype(np.int64)
        eligible = (z > .1) & (z < 4) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        gi = np.flatnonzero(eligible)
        dep = d_full[uv[gi, 1], uv[gi, 0]]
        observed = np.column_stack([(uv[gi, 0] - cam["cx"]) * dep / cam["fx"],
                                    (uv[gi, 1] - cam["cy"]) * dep / cam["fy"], dep])
        support = (dep > 0) & (dep < 4) & (np.linalg.norm(observed - camera_xyz[gi], axis=1) < .02)
        gi = gi[support]
        if len(gi):
            gs[gi] += 1
            ray_cam = camera_xyz[gi] / z[gi, None]
            dr = ray_cam @ pose[:3, :3].T
            rays = np.column_stack([np.broadcast_to(pose[:3, 3], dr.shape), dr]).astype(np.float32)
            h = caster.cast_rays(o3d.core.Tensor(rays), nthreads=4)
            hz = h["t_hit"].numpy()
            hid = h["primitive_ids"].numpy().astype(np.int64)
            hc = np.full(len(gi), -1, dtype=np.int32)
            finite = np.isfinite(hz)
            hc[finite] = tri_classes[hid[finite]]
            gw[gi] += hc == window_c
            gbehind[gi] += (hc == window_c) & (z[gi] - hz > .1)
            gother[gi] += finite & (hc != window_c)
            galign[gi] += finite & (np.abs(z[gi] - hz) <= .1)
        if seq % 50 == 0:
            print(scene, "FRAMES", seq + 1, "/", len(config["frame_ids"]), flush=True)
    summary = {}
    for name, values in accumulated.items():
        totals = np.sum(values["counts"], axis=0)
        summary[name] = dict(zip(["interior_pixel_frames", "missing", "over4", "accepted", "accepted_behind10cm", "accepted_front10cm"], map(int, totals)))
        summary[name]["accepted_depth_residual"] = stats(np.concatenate(values["accepted_residual"]))
        summary[name]["raw_valid_depth_residual"] = stats(np.concatenate(values["raw_valid_residual"]))
    trace = dict(sampled_fixed_surface_points=len(idx), over10cm_from_GT=len(ghost),
                 depth_supported_within2cm=int((gs > 0).sum()),
                 support_ray_hits_window_any=int((gw > 0).sum()),
                 support_behind_window_by10cm_any=int((gbehind > 0).sum()),
                 window_majority_of_supports=int(((gs > 0) & (gw > gs / 2)).sum()),
                 nonwindow_support_any=int((gother > 0).sum()),
                 ray_depth_near_GT_any=int((galign > 0).sum()),
                 caveat="GT-cleaned mesh is a diagnostic reference; window label does not identify specular vs transmissive optics. >10cm is not automatically a false surface.")
    save(out / "raw_depth_regions.json", dict(scene=scene, pixel_stride=4, frames=len(config["frame_ids"]),
          ROI="GT first-hit z in (0.2,4)m; per-class 3x3 erosion at stride4; GT depth range <8cm in 3x3",
          region_summary=summary, frame_records=per_frame, ghost_support_trace=trace,
          GT_used_only_for_diagnosis=True, map_mutated=False, elapsed_s=time.time() - started))
    np.savez_compressed(out / "ghost_source_trace.npz", xyz=ghost, support=gs, window=gw,
                        behind_window=gbehind, other=gother, near_GT_ray=galign)
    # Preserve representative RGB/depth/reference maps for independent inspection.
    selected = sorted(examples, reverse=True)
    selected_fids = []
    for _, fid, _ in selected:
        if all(abs(fid - f) >= 100 for f in selected_fids):
            selected_fids.append(fid)
        if len(selected_fids) == 3:
            break
    selected_fids += [f for f in ([0, 1190, 2010, 2370] if scene == "scene0011_00" else [0, 1420, 3380]) if f not in selected_fids]
    for fid in selected_fids:
        d = cv2.imread(str(processed / "depth" / (str(fid) + ".png")), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        pose = poses[fid]
        v1, u1 = np.mgrid[:480, :640]
        rc = np.stack([(u1 - cam["cx"]) / cam["fx"], (v1 - cam["cy"]) / cam["fy"], np.ones_like(u1)], axis=-1)
        directions = rc @ pose[:3, :3].T
        rays = np.concatenate([np.broadcast_to(pose[:3, 3], directions.shape), directions], axis=-1).astype(np.float32)
        h = caster.cast_rays(o3d.core.Tensor(rays), nthreads=4)
        rd = h["t_hit"].numpy()
        hid = h["primitive_ids"].numpy().astype(np.int64)
        cl = np.full(rd.shape, -1, np.int32)
        good = np.isfinite(rd)
        cl[good] = tri_classes[hid[good]]
        rgb = cv2.imread(str(R / "inputs" / scene / "results" / f"frame{fid:06d}.png"))
        cv2.imwrite(str(out / f"frame{fid}_rgb.jpg"), rgb)
        np.savez_compressed(out / f"frame{fid}_depth.npz", depth=d, reference_depth=rd, reference_class=cl,
                            class_names=np.array(name_list), pose=pose, K=K)
    print(scene, "REGIONS", json.dumps(summary), flush=True)
    print(scene, "GHOST", json.dumps(trace), flush=True)
    print(scene, "DONE", time.time() - started, flush=True)


for scene in ["scene0011_00", "scene0608_00"]:
    scene_audit(scene)
