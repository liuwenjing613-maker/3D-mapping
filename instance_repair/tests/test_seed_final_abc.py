import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from instance_repair.seed_final_abc import (
    filter_final, passes, finalize, validate_selected, read, save, sha, POLICY)
from instance_repair.seed_boundary import POLICY as BOUNDARY_POLICY
from instance_repair.seed_abc_projection import correspondence, source_cloud, cloud_visibility


def triple(a="A", b="B", c="C"):
    return dict(A=a, B=b, C=c, target_frame=0, mask_hashes={m: m for m in (a, b, c)},
                view_choice=dict(frame=0, visible_fractions={m: .8 for m in (a, b, c)}),
                metric=dict(areas=dict(A=100, B=50, C=50, union_BC=100,
                                      covered_A=100, intersection_BC=0)))


def fixture():
    seeds = {m: dict(id=m, frame=0, mask=m+".png", mask_sha256=m, digest=m,
                     support_sha256=m, aliases=[dict(semantic_family="window")])
             for m in ("A", "B", "C", "alias")}
    manifest = dict(status="automatic_v2_complete", unrelated=dict(keep=[1, 2]),
                    groups=[dict(id="G"+m, primary=copy.deepcopy(seeds[m]),
                                 auxiliary_ids=["alias"], original_track_id=i+4)
                            for i, m in enumerate(("B", "A", "alias"))])
    semantics = {m: dict(top_class="sofa", exclude=False) for m in seeds}
    search = dict(status="PASS", anchors=[dict(A="A", exhaustive=True, tested_pairs=1,
                                             possible_pairs=1, evaluations=[triple()],
                                             duplicate_class=["A", "alias"])])
    return manifest, search, semantics, seeds


class FinalABC(unittest.TestCase):
    def test_current_registry_requires_final_handoff(self):
        root = Path(__file__).resolve().parents[2]
        registry = read(root / "instance_repair/configs/current_auto_pipeline.json")
        self.assertEqual(registry["final_seed_policy"], POLICY)
        self.assertTrue(registry["final_seed_handoff_required"])
        self.assertEqual(registry["final_seed_entry"],
                         "python -B -m instance_repair.seed_final_abc finalize")
        self.assertEqual(registry["tracking_handoff_guard"],
                         "python -B -m instance_repair.seed_final_abc validate-selected")
        config = read(root / registry["final_seed_config"])
        self.assertFalse(config["add_BC"])
        self.assertFalse(config["restore_candidates"])
        self.assertFalse(config["rerun_downstream"])
        self.assertFalse(config["new_boundary_filter"])

    def test_exact_A_only_order_metadata_no_child_promotion_or_alias_deletion(self):
        args = fixture()
        before = copy.deepcopy(args)
        output, audit = filter_final(*args)
        self.assertEqual(output["groups"], [args[0]["groups"][0], args[0]["groups"][2]])
        self.assertEqual(output["unrelated"], args[0]["unrelated"])
        self.assertEqual(audit["removed_A"], ["A"])
        self.assertEqual(audit["added"], [])
        self.assertEqual(audit["restored"], [])
        self.assertEqual(args, before)

    def test_window_and_blind_unchanged_but_cabinet_and_window_roi_sofa_are_not_exempt(self):
        for label in ("windowpane", "blind", "cabinet", "sofa"):
            with self.subTest(label=label):
                args = fixture()
                args[2]["A"]["top_class"] = label
                output, audit = filter_final(*args)
                self.assertEqual(audit["removed_A"], [] if label in ("windowpane", "blind") else ["A"])
                if label in ("windowpane", "blind"):
                    self.assertEqual(output["groups"], args[0]["groups"])

    def test_thresholds_and_largest_A(self):
        self.assertTrue(passes(triple()))
        for edits, visibility in [
            ({"covered_A": 70}, .8),
            ({"A": 70, "covered_A": 70}, .8),
            ({"B": 50, "C": 55, "union_BC": 100, "intersection_BC": 5}, .8),
            ({"B": 101, "C": 1, "union_BC": 102, "covered_A": 100}, .8),
            ({}, .799999),
            ({"A": 0, "covered_A": 0}, .8),
        ]:
            rec = triple()
            rec["metric"]["areas"].update(edits)
            rec["view_choice"]["visible_fractions"]["A"] = visibility
            self.assertFalse(passes(rec), edits)

    def test_multiple_pairs_do_not_block_removal(self):
        args = fixture()
        args[3]["D"] = dict(args[3]["C"], id="D", mask_sha256="D")
        args[2]["D"] = dict(top_class="sofa", exclude=False)
        anchor = args[1]["anchors"][0]
        anchor["evaluations"].append(triple(c="D"))
        anchor["tested_pairs"] = anchor["possible_pairs"] = 2
        self.assertEqual(filter_final(*args)[1]["removed_A"], ["A"])

    def test_fail_closed_on_early_stage_missing_semantics_stale_masks_incomplete_search(self):
        changes = [
            lambda a: a[0].update(status="boundary_screened_needs_semantic_and_v2_selection"),
            lambda a: a[2]["A"].pop("top_class"),
            lambda a: a[3]["A"].update(mask_sha256="stale"),
            lambda a: a[1]["anchors"][0].update(exhaustive=False),
            lambda a: a[1]["anchors"][0]["evaluations"][0]["mask_hashes"].update(B="stale"),
        ]
        for change in changes:
            args = fixture()
            change(args)
            with self.assertRaises(ValueError):
                filter_final(*args)

    def test_idempotent_guard_and_empty_final_list(self):
        args = fixture()
        output, _ = filter_final(*args)
        with self.assertRaises(ValueError):
            filter_final(output, *args[1:])
        args[0]["groups"] = []
        self.assertEqual(filter_final(*args)[0]["groups"], [])

    def test_projection_identity_and_occluded_source(self):
        camera = dict(fx=10., fy=10., cx=4.5, cy=4.5)
        depth = np.ones((10, 10))
        mask = np.zeros((10, 10), bool)
        mask[3:7, 3:7] = True
        pose = np.eye(4)
        index, known = correspondence(depth, depth, pose, pose, camera)
        np.testing.assert_array_equal(mask.ravel()[index] & known, mask)
        cloud = source_cloud(mask, depth, pose, camera)
        self.assertEqual(cloud_visibility(cloud, depth, pose, camera), 1.)
        self.assertEqual(cloud_visibility(cloud, depth * 2, pose, camera), 0.)

    def test_registered_handoff_end_to_end_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (680, 1200)
            camera = dict(fx=600., fy=600., cx=599.5, cy=339.5,
                          width=1200, height=680, depth_png_units_per_meter=1000.)
            np.savetxt(root/"traj.txt", np.eye(4))
            Image.fromarray(np.full(shape, 1000, np.uint16)).save(root/"depth0.png")
            seeds, supports = [], []
            for i, mid in enumerate(("A", "B", "C")):
                mask = np.zeros(shape, bool)
                mask[100:110, 100:110 if mid=="A" else 105] = True
                if mid == "C":
                    mask[:] = False
                    mask[100:110, 105:110] = True
                path = root/(mid+".png")
                Image.fromarray(mask.astype(np.uint8)*255).save(path)
                support = np.arange(100 if mid=="A" else 50, dtype=np.int64)
                if mid == "C":
                    support += 50
                supports.append(support)
                seeds.append(dict(id=mid, index=i, frame=0, area=int(mask.sum()), mask=path.name,
                    mask_sha256=sha(path),
                    digest=hashlib.sha256((0).to_bytes(4,"little")+np.packbits(mask.ravel()).tobytes()).hexdigest(),
                    support_count=len(support), support_sha256=hashlib.sha256(support.tobytes()).hexdigest(),
                    representative=dict(model=0), noise_decision=dict(exclude=False),
                    aliases=[dict(model=0,source_box=[0,0,1200,680])]))
            grouping = dict(rule=dict(scene="synthetic", boundary_policy=BOUNDARY_POLICY),
                seeds=seeds, groups=[dict(id="G"+s["id"],primary=i) for i,s in enumerate(seeds)])
            save(root/"grouping.json", grouping)
            np.savez(root/"supports.npz", indices=np.concatenate(supports), offsets=np.array([0,100,150,200]))
            np.savez(root/"pairs.npz", intersection=np.array([[100,50,50],[50,50,0],[50,0,50]]))
            np.savez(root/"frame000000_visible.npz",vertex_index=np.arange(100),uv=np.zeros((100,2)))
            save(root/"geometry.json",dict(frame_ids=[0],frame_count=1,no_future_geometry=True))
            save(root/"mapping.json",dict(camera=camera,source=dict(scene_root=str(root),
                trajectory="traj.txt",depth_pattern="depth{frame}.png")))
            save(root/"semantics.json",dict(records={s["id"]:dict(top_class="sofa",exclude=False) for s in seeds},
                 source_hashes={str(root/"grouping.json"):sha(root/"grouping.json")}))
            save(root/"manifest.json",dict(status="automatic_v2_complete",
                groups=[dict(id="GA",primary=seeds[0],auxiliary_ids=[],original_track_id=17)]))
            context = {k:str(root/v) for k,v in dict(grouping="grouping.json",supports="supports.npz",
                pairs="pairs.npz",geometry="geometry.json",mapping_config="mapping.json",
                mask_root=".",visible_root=".",semantics="semantics.json").items()}
            save(root/"context.json",context)
            save(root/"geometry.json",dict(frame_ids=[1],frame_count=1,no_future_geometry=True))
            with self.assertRaises(ValueError):
                finalize(root/"manifest.json",root/"context.json",root/"future_rejected")
            save(root/"geometry.json",dict(frame_ids=[0],frame_count=1,no_future_geometry=True))
            result = finalize(root/"manifest.json",root/"context.json",root/"final")
            self.assertEqual(result["removed_A"],["A"])
            self.assertEqual(result["output_count"],0)
            self.assertEqual(read(root/"final/seed_manifest.json")["groups"],[])
            with self.assertRaises(ValueError):
                validate_selected(root/"manifest.json",root/"final/receipt.json")
            final = root/"final/seed_manifest.json"
            changed = read(final)
            changed["groups"] = read(root/"manifest.json")["groups"]
            save(final,changed)
            with self.assertRaises(ValueError):
                validate_selected(final,root/"final/receipt.json")


if __name__ == "__main__":
    unittest.main()
