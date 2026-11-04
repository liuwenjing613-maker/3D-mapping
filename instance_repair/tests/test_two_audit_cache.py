"""Numerical/causal/provenance tests for shared raw observations, not repair."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from instance_repair.two_audit_cache import build_cache, ObservationCache
from instance_repair.two_audit_projection import ProjectionSettings, project_frame
from instance_repair.two_audit_sources import MappingSource, geometry_identity, sha256_file


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, count=2, per_frame=True, background_only=False):
    raw = root / "original"
    (raw / "scene/results").mkdir(parents=True)
    (raw / "masks").mkdir()
    mask = np.array([[1, 1, 0, 0, 2], [1, 1, 0, 3, 2], [0, 0, 4, 0, 0]], np.uint16)
    if background_only:
        mask[:] = 0
    depth = np.full((3, 5), 1000, np.uint16)
    depth[1, 3] = 0
    depth[2, 2] = 11000
    records, metadata = [], []
    for fid in (0, 5):
        mpath = raw / "masks" / f"frame{fid:06d}.png"
        Image.fromarray(mask).save(mpath)
        Image.fromarray(depth).save(raw / "scene/results" / f"depth{fid:06d}.png")
        digest = sha256_file(mpath)
        metadata.append(dict(frame=fid, mask_sha256=digest, shape=list(mask.shape)))
        for mid in np.unique(mask):
            if not mid:
                continue
            y, x = np.where(mask == mid)
            valid = (depth[y, x] > 0) & (depth[y, x] < 10000)
            records.append(dict(observation_id=f"Replica/test/f{fid:06d}/m{mid:03d}",
                                frame_id=fid, mask_local_id=int(mid),
                                source_mask_sha256=digest, pixel_count=len(y),
                                projectable_pixel_count=int(valid.sum()),
                                bbox_xyxy_exclusive=[int(x.min()), int(y.min()),
                                                     int(x.max()+1), int(y.max()+1)]))
    (raw / "masks/frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    poses = np.tile(np.eye(4), (6, 1, 1))
    poses[5, 0, 3] = .2
    intrinsics = np.tile(np.eye(3), (6, 1, 1))
    intrinsics[5, 0, 0] = 2
    np.savetxt(raw / "scene/traj.txt", poses.reshape(6, 16))
    np.savetxt(raw / "scene/intrinsics.txt", intrinsics.reshape(6, 9))
    camera = dict(width=5, height=3, fx=1., fy=1., cx=0., cy=0.,
                  depth_png_units_per_meter=1000)
    if per_frame:
        camera["per_frame_intrinsics"] = "intrinsics.txt"
    config = dict(dataset="Replica", scene="test", camera=camera,
                  frame_selection=dict(start=0, stop_exclusive=6, stride=5),
                  mask=dict(background_id=0, provenance="CropFormer"),
                  source=dict(scene_root=str(raw / "scene"), mask_root=str(raw / "masks"),
                              trajectory="traj.txt", depth_pattern="results/depth{frame:06d}.png",
                              mask_pattern="frame{frame:06d}.png", mask_metadata="frames.jsonl"),
                  rules=dict(use_future_frames=False))
    original_path = raw / "config.json"
    dump(original_path, config)
    mapping = root / "mapping"
    selected = [0, 5][:count]
    mapping_config = json.loads(json.dumps(config))
    mapping_config["frame_selection"]["stop_exclusive"] = selected[-1] + 1
    dump(mapping / "config.json", mapping_config)
    dump(mapping / "mapping_plan.json",
         dict(original_config=str(original_path), raw_cutoff=selected[-1],
              target_processed_frames=count, prior_processed_frames=0))
    dump(mapping / "mapping_complete.json",
         dict(status="PASS", frame_ids=selected, frame_count=count,
              last_frame_id=selected[-1], GT_used=False, future_inputs_used=False,
              future_baseline_used=False, current_checkpoint_repair_applied=False,
              native_shared_geometry=True, geometry_incremental=True, repair_applied=False))
    (mapping / "association").mkdir()
    (mapping / "association/observations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records if row["frame_id"] in selected),
        encoding="utf-8")
    xyz = np.array([[0, 0, 1], [2, 0, 1], [4, 0, 1], [0, 0, 1.2],
                    [0, 0, -1], [3, 1, 1], [2, 2, 11], [8, 0, 1],
                    [1, 0, 1.01]], np.float32)
    (mapping / "baseline/native").mkdir(parents=True)
    np.savez_compressed(mapping / "baseline/native/instance_surface.npz",
                        xyz_m=xyz, instance_id=np.full(len(xyz), 987, np.int32))
    # Existing repaired labels are deliberately unreadable; the audit must not
    # accidentally consume frame_support or identity journals.
    (mapping / "baseline/frame_support").mkdir()
    (mapping / "baseline/frame_support/f000000.npz").write_bytes(b"not original observations")
    return mapping, xyz


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="two-audit-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, mapping, name="cache", mode="offline", checkpoint=2):
        with redirect_stdout(io.StringIO()):
            return build_cache(mapping, self.root / name, mode=mode, checkpoint=checkpoint)

    def test_visibility_is_depth_checked_and_includes_background(self):
        mapping, xyz = fixture(self.root)
        source = MappingSource(mapping, "offline", 2)
        frame = source.load_frame(0, 10.)
        pack, stats = project_frame(frame, xyz, ProjectionSettings(point_chunk_size=2))
        np.testing.assert_array_equal(pack["visible_surface_indices"], [0, 1, 2, 8])
        np.testing.assert_array_equal(pack["visible_mask_local_ids"], [1, 0, 2, 1])
        np.testing.assert_array_equal(pack["mask_surface_indices"], [0, 8, 2])
        np.testing.assert_array_equal(pack["mask_offsets"], [0, 2, 3, 3, 3])
        self.assertEqual(stats["background_visible_points"], 1)

    def test_per_frame_intrinsics_and_pose_are_used(self):
        mapping, xyz = fixture(self.root)
        source = MappingSource(mapping, "offline", 2)
        frame = source.load_frame(5, 10.)
        self.assertEqual(frame.intrinsic[0, 0], 2)
        self.assertAlmostEqual(frame.pose[0, 3], .2)
        pack, _ = project_frame(frame, xyz, ProjectionSettings())
        self.assertNotIn(2, pack["visible_surface_indices"])  # u=7.6 outside width 5

    def test_static_camera_contract(self):
        mapping, _ = fixture(self.root, per_frame=False)
        source = MappingSource(mapping, "offline", 2)
        np.testing.assert_array_equal(source.load_frame(5, 10.).intrinsic, np.eye(3))
        self.assertEqual(len(source.prefix_sources), 2)  # poses + mask metadata

    def test_complete_cache_preserves_zero_support_masks_and_never_map_ids(self):
        mapping, xyz = fixture(self.root)
        before = sha256_file(mapping / "baseline/native/instance_surface.npz")
        result = self.build(mapping)
        cache = ObservationCache(self.root / "cache")
        self.assertEqual(cache.validate(verify_sources=True)["status"], "PASS")
        self.assertEqual(result["totals"]["observations"], 8)
        rows = cache.by_frame[0]
        empty = next(row for row in rows if row["mask_local_id"] == 3)
        self.assertEqual(empty["surface_support_count"], 0)
        self.assertIn("no_valid_depth", empty["quality_flags"])
        self.assertFalse(any("instance_id" in row or "persistent_id" in row for row in rows))
        self.assertEqual(before, sha256_file(mapping / "baseline/native/instance_surface.npz"))
        gid = cache.require_geometry(xyz)
        np.testing.assert_array_equal(cache.support(rows[0]["observation_key"], geometry_id=gid),
                                      [0, 8])
        self.assertIn(1, cache.visible(0, geometry_id=gid))  # background still visible
        with self.assertRaises(TypeError):
            cache.visible(0)  # geometry binding is mandatory

    def test_online_never_loads_future_masks_depth_or_pose(self):
        mapping, _ = fixture(self.root, count=1)
        for path in [self.root / "original/masks/frame000005.png",
                     self.root / "original/scene/results/depth000005.png"]:
            path.unlink()
        pose = self.root / "original/scene/traj.txt"
        first = pose.read_bytes().splitlines(keepends=True)[0]
        pose.write_bytes(first + b"UNREADABLE FUTURE POSE\n")
        self.build(mapping, mode="online-v2", checkpoint=1)
        cache = ObservationCache(self.root / "cache")
        self.assertEqual(cache.manifest["frame_ids"], [0])
        self.assertEqual(cache.validate(verify_sources=True)["status"], "PASS")
        source = MappingSource(mapping, "online-v2", 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            source.load_frame(5, 10.)

    def test_offline_requires_full_experimental_sequence(self):
        mapping, _ = fixture(self.root, count=1)
        with self.assertRaisesRegex(ValueError, "full experimental sequence"):
            self.build(mapping, checkpoint=1)

    def test_wrong_checkpoint_and_future_receipt_are_rejected(self):
        mapping, _ = fixture(self.root)
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.build(mapping, checkpoint=1)
        path = mapping / "mapping_complete.json"
        receipt = json.loads(path.read_text())
        receipt["future_inputs_used"] = True
        dump(path, receipt)
        with self.assertRaisesRegex(ValueError, "future_inputs_used"):
            self.build(mapping)

    def test_geometry_order_is_part_of_identity(self):
        mapping, xyz = fixture(self.root)
        self.build(mapping)
        cache = ObservationCache(self.root / "cache")
        reordered = xyz[::-1].copy()
        self.assertNotEqual(geometry_identity(xyz)["geometry_id"],
                            geometry_identity(reordered)["geometry_id"])
        with self.assertRaisesRegex(ValueError, "recompute"):
            cache.require_geometry(reordered)
        with self.assertRaisesRegex(ValueError, "another geometry"):
            cache.visible(0, geometry_id=geometry_identity(reordered)["geometry_id"])

    def test_original_identity_is_stable_but_projection_rebuilt_for_new_geometry(self):
        mapping, xyz = fixture(self.root)
        self.build(mapping, name="one")
        surface = mapping / "baseline/native/instance_surface.npz"
        np.savez_compressed(surface, xyz_m=xyz[::-1].copy())
        self.build(mapping, name="two")
        a, b = ObservationCache(self.root / "one"), ObservationCache(self.root / "two")
        self.assertEqual(set(a.observations), set(b.observations))
        self.assertNotEqual(a.geometry_id, b.geometry_id)
        self.assertNotEqual(a.manifest["cache_id"], b.manifest["cache_id"])
        row = a.by_frame[0][0]
        sa = a.support(row["observation_key"], geometry_id=a.geometry_id)
        sb = b.support(row["observation_key"], geometry_id=b.geometry_id)
        self.assertEqual(set(sa), set(len(xyz) - 1 - sb))

    def test_modified_original_mask_hash_is_rejected_without_completion_marker(self):
        mapping, _ = fixture(self.root)
        path = self.root / "original/masks/frame000000.png"
        Image.fromarray(np.zeros((3, 5), np.uint16)).save(path)
        with self.assertRaisesRegex(ValueError, "mask hash"):
            self.build(mapping)
        self.assertFalse((self.root / "cache/manifest.json").exists())
        self.assertEqual(json.loads((self.root / "cache/status.json").read_text())["status"],
                         "FAILED")

    def test_repaired_mask_substitution_is_rejected_even_with_same_pixels(self):
        mapping, _ = fixture(self.root)
        substitute = self.root / "repaired"
        substitute.mkdir()
        for fid in (0, 5):
            (substitute / f"frame{fid:06d}.png").write_bytes(
                (self.root / "original/masks" / f"frame{fid:06d}.png").read_bytes())
        (substitute / "frames.jsonl").write_bytes(
            (self.root / "original/masks/frames.jsonl").read_bytes())
        config = json.loads((mapping / "config.json").read_text())
        config["source"]["mask_root"] = str(substitute)
        dump(mapping / "config.json", config)
        with self.assertRaisesRegex(ValueError, "original mask source"):
            self.build(mapping)

    def test_cache_corruption_and_output_overwrite_are_rejected(self):
        mapping, _ = fixture(self.root)
        self.build(mapping)
        with self.assertRaises(FileExistsError):
            self.build(mapping)
        pack = self.root / "cache/frames/f000000.npz"
        pack.write_bytes(pack.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "support hash mismatch"):
            ObservationCache(self.root / "cache").validate()

    def test_background_only_frame_has_visibility_and_no_invented_observations(self):
        mapping, _ = fixture(self.root, background_only=True)
        result = self.build(mapping)
        self.assertEqual(result["totals"]["observations"], 0)
        cache = ObservationCache(self.root / "cache")
        pack = cache.read_frame(0, geometry_id=cache.geometry_id)
        self.assertEqual(len(pack["mask_surface_indices"]), 0)
        self.assertEqual(len(pack["visible_surface_indices"]), 4)
        self.assertEqual(cache.validate()["status"], "PASS")

    def test_changed_source_after_projection_detected(self):
        mapping, _ = fixture(self.root)
        source = MappingSource(mapping, "offline", 2)
        source.load_frame(0, 10.)
        path = self.root / "original/scene/results/depth000000.png"
        Image.fromarray(np.full((3, 5), 2000, np.uint16)).save(path)
        with self.assertRaisesRegex(ValueError, "Source content changed"):
            source.verify_unchanged()


if __name__ == "__main__":
    unittest.main()
