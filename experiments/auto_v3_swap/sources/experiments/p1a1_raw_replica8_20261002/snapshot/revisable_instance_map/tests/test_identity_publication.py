"""End-to-end revision publication with overlapping same-frame raw regions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(PACKAGE / "tools"))
from revisable_instance_map.association import OnlineVoxelAssociator
from revisable_instance_map.observations import RawInstanceObservation
from revisable_instance_map.surface_evidence import frame_instance_keys, reduce_surface_votes
from rebuild_p0_from_regions import rebuild_surface
from run_baseline_association import sha256_file


class IdentityPublicationTest(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"
        source.mkdir()
        config, catalog = root / "config.json", root / "observations.jsonl"
        config.write_text(json.dumps({"frame_selection": {"start": 0, "stop_exclusive": 10, "stride": 5}}))
        records, manifest, keys = [], [], []
        state = OnlineVoxelAssociator(association_mode="probabilistic")
        xyz, rgb = np.zeros((3, 3), np.float32), np.ones((3, 3), np.float32)
        for frame in (0, 5):
            for local in (1, 2):
                observation = RawInstanceObservation(f"f{frame}/m{local}", frame, local, "0" * 64,
                                                      1, 1, (0, 0, 1, 1), None, None, None)
                records.append(observation.__dict__)
                state._commit(1, observation, ((0, 0, 0),))
            points, local_ids = np.array([0, 0, 1, 2], np.int32), np.array([1, 2, 1, 2], np.int32)
            support = source / f"f{frame}.npz"
            np.savez_compressed(support, frame_id=[frame], surface_point_index=points, mask_local_id=local_ids,
                source_mask_sha256=["0" * 64], pixel_stride=[2], max_surface_distance_m=[.015])
            manifest.append(dict(frame_id=frame, support_file=str(support), support_sha256=sha256_file(support)))
            keys.append(frame_instance_keys(points, local_ids, np.array([-1, 1, 1]), 2))
        state.last_processed_frame_id = 5
        state.map_version = 2
        catalog.write_text("".join(json.dumps(row) + "\n" for row in records))
        support_manifest = source / "frame_region_support_manifest.jsonl"
        support_manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest))
        unique, votes = np.unique(np.concatenate(keys), return_counts=True)
        evidence = reduce_surface_votes(unique, votes, 3, 2)
        np.savez_compressed(source / "surface_evidence.npz", xyz_m=xyz, rgb=rgb, **evidence)
        report = dict(observation_catalog_sha256=sha256_file(catalog), config_sha256=sha256_file(config),
            frame_region_support_manifest_sha256=sha256_file(support_manifest),
            surface_evidence_sha256=sha256_file(source / "surface_evidence.npz"),
            tsdf_surface_sha256="geometry", pixel_stride=2, max_surface_distance_m=.015,
            min_confirmed_votes=2, min_confirmed_ratio=.67, scene_id="synthetic", frame_count=2)
        (source / "materialization_report.json").write_text(json.dumps(report))
        return source, config, catalog, state, evidence

    def publish(self, root, source, config, catalog, state):
        checkpoint, assignments = root / "checkpoint.npz", root / "assignments.jsonl"
        state.save_checkpoint(checkpoint)
        assignments.write_text("".join(json.dumps(dict(observation_id=key, frame_id=entry.frame_id,
            instance_id=entry.instance_id, assignment_version=entry.assignment_version)) + "\n"
            for key, entry in state.observation_support.items()))
        return rebuild_surface(source, catalog, assignments, checkpoint, config, root / "published", 2)

    def test_revision_recounts_raw_regions_and_round_trip_restores_p0(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, catalog, state, original = self.fixture(root)
            state.reassign_observations({"f0/m1": 2})
            report = self.publish(root, source, config, catalog, state)
            self.assertEqual(report["map_version"], 3)
            with np.load(root / "published/surface_evidence.npz") as data:
                # Overlapping m2 keeps the original frame's A vote at point0.
                self.assertEqual(data["top1_votes"][0], 2)
                self.assertEqual(data["top2_votes"][0], 1)
                self.assertEqual(data["map_version"][0], state.map_version)
            state.reassign_observations({"f0/m1": 1})
            self.publish(root, source, config, catalog, state)
            with np.load(root / "published/surface_evidence.npz") as data:
                for key, value in original.items():
                    np.testing.assert_array_equal(data[key], value)

    def test_cache_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, catalog, state, _ = self.fixture(root)
            np.savez(source / "f0.npz", garbage=[1])
            with self.assertRaisesRegex(ValueError, "modified"):
                self.publish(root, source, config, catalog, state)

    def test_mismatched_assignment_version_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, catalog, state, _ = self.fixture(root)
            self.publish(root, source, config, catalog, state)
            state.reassign_observations({"f0/m1": 2})
            state.save_checkpoint(root / "checkpoint.npz")
            with self.assertRaisesRegex(ValueError, "differ"):
                rebuild_surface(source, catalog, root / "assignments.jsonl", root / "checkpoint.npz",
                                config, root / "rejected", 2)


if __name__ == "__main__":
    unittest.main()
