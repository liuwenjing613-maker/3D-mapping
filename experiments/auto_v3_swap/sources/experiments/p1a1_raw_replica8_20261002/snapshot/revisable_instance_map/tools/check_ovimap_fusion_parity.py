#!/usr/bin/env python3
"""Compare all refined regions to official OVI-MAP frameToSegmentsCropFormer.

Run inside the native OVI container with PYTHONPATH including its scripts.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from utils.common_scannet_nyu import SegmentsGenerator


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw-mask-dir", type=Path, required=True)
    p.add_argument("--geometry-dir", type=Path, required=True)
    p.add_argument("--refined-dir", type=Path, required=True)
    p.add_argument("--scene-root", type=Path, default=Path("/datasets/Replica/room0"))
    a=p.parse_args()
    expected=defaultdict(list)
    for line in (a.refined_dir/"segments.jsonl").open(encoding="utf-8"):
        row=json.loads(line)
        expected[row["frame_id"]].append(row)
    frame_rows={json.loads(line)["frame"]:json.loads(line) for line in (a.refined_dir/"frames.jsonl").open(encoding="utf-8")}
    K=np.array([[600,0,599.5],[0,600,339.5],[0,0,1]],dtype=np.float32)
    generator=SegmentsGenerator(None,None,None,geometrics_folder=str(a.geometry_dir))
    started=time.perf_counter()
    checked=0
    backgrounds=0
    for index,frame_id in enumerate(range(0,2000,5),1):
        depth_path=a.scene_root/"results"/("depth%06d.png"%frame_id)
        raw_path=a.raw_mask_dir/("frame%06d.png"%frame_id)
        depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED).astype(np.float32)/6553.5
        crop=cv2.imread(str(raw_path),cv2.IMREAD_UNCHANGED).astype(np.int32)
        if depth.shape!=(680,1200) or crop.shape!=(680,1200):
            raise ValueError("Bad frame shape at %d"%frame_id)
        official=generator.frameToSegmentsCropFormer(depth,K,np.eye(4,dtype=np.float32),frame_id,crop)
        things=[x for x in official if x.is_thing]
        bg=[x for x in official if not x.is_thing]
        ours=expected[frame_id]
        if len(things)!=len(ours) or len(bg)!=frame_rows[frame_id]["background_regions"]:
            raise AssertionError("Region count mismatch at %d: official %d, ours %d"%(frame_id,len(things),len(ours)))
        for position,(official_region,our_region) in enumerate(zip(things,ours)):
            if (int(official_region.instance_label)!=our_region["cropformer_id"]
                    or len(official_region.points)!=our_region["pixel_count"]
                    or abs(float(official_region.overlap_ratio)-our_region["overlap_ratio"])>1e-6):
                raise AssertionError("Region mismatch at frame %d position %d"%(frame_id,position))
        checked+=len(things)
        backgrounds+=len(bg)
        if index%50==0:
            print(json.dumps({"frames":index,"regions":checked,"elapsed_s":round(time.perf_counter()-started,1)}),flush=True)
    report={"status":"PASS","frames":400,"positive_regions":checked,"background_regions":backgrounds,"official_source":"scripts/utils/common_scannet_nyu.py:frameToSegmentsCropFormer","matched_fields":["order","cropformer_id","pixel_count","overlap_ratio","background_region_count"],"seconds":time.perf_counter()-started}
    path=a.refined_dir/"official_fusion_parity.json"
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
