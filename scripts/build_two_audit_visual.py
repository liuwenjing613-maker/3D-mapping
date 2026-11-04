"""Build the confirmed visual comparison page from immutable Replica audit evidence."""
import argparse, json, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from instance_repair.two_audit_cache import ObservationCache, write_json
from instance_repair.two_audit_sources import file_record, verify_record
from scripts.render_two_audit_anomaly_review import PALETTE, choose_cards

def sample(indices, limit):
    return indices[np.linspace(0,len(indices)-1,limit,dtype=int)] if len(indices)>limit else indices

def build(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).absolute()
    if output.exists(): raise FileExistsError(output)
    validation=json.loads((root/"validation.json").read_text())
    assert validation["status"]=="PASS" and validation["dataset"]=="Replica"
    output.mkdir(parents=True); (output/"cases").mkdir()
    results=[]; sources=[]; artifacts=[]
    for scene in ("room0","office2"):
        audit=root/"instance_anomalies"/scene
        manifest=json.loads((audit/"manifest.json").read_text())
        for name,sha in manifest["artifact_sha256"].items():
            record=file_record(audit/name); assert record["sha256"]==sha; sources.append(record)
        cache=ObservationCache(manifest["cache_root"])
        native=Path(cache.manifest["mapping_root"])/"baseline/native/instance_surface.npz"
        nr=file_record(native)
        locked=[r for r in cache.manifest["locked_sources"] if r["resolved_path"]==str(native.resolve())]
        assert len(locked)==1 and locked[0]["sha256"]==nr["sha256"]; sources.append(nr)
        with np.load(native,allow_pickle=False) as z: xyz=z["xyz_m"]; ids=z["instance_id"]
        cache.require_geometry(xyz)
        cards=choose_cards(json.loads((audit/"candidate_pairs.json").read_text()))
        render=json.loads((audit/"review/render_validation.json").read_text())
        assert render["status"]=="PASS"
        figures=defaultdict(list)
        for f in render["figures"]: figures[f["pair_key"]].append(f)
        neighbors=defaultdict(set)
        for card in cards:
            a,b=card["pair"]["track_ids"];neighbors[a].add(b);neighbors[b].add(a)
        colors={}
        palette=list(PALETTE)
        for tid in sorted(neighbors,key=lambda t:(-len(neighbors[t]),t)):
            used={colors[t] for t in neighbors[tid] if t in colors}
            ci=next((i for i in range(len(palette)) if i not in used),None)
            if ci is None:
                from colorsys import hsv_to_rgb
                ci=len(palette);palette.append(tuple(int(255*x) for x in hsv_to_rgb((ci*.61803398875)%1,.78,1)))
            colors[tid]=ci
        track_root=root/"stable_tracks"/scene
        tm=json.loads((track_root/"manifest.json").read_text())
        assert tm["cache_id"]==cache.manifest["cache_id"] and tm["geometry_id"]==cache.geometry_id
        sources.append(file_record(track_root/"manifest.json"))
        tr=file_record(track_root/"stable_tracks.json")
        assert tr["sha256"]==tm["artifact_sha256"]["stable_tracks.json"];sources.append(tr)
        tracks={t["track_id"]:t for t in json.loads((track_root/"stable_tracks.json").read_text())}
        needed={tid for card in cards for tid in card["pair"]["track_ids"]}
        accum={tid:np.zeros(len(xyz),bool) for tid in needed}
        members=defaultdict(list)
        for tid in needed:
            t=tracks[tid]
            assert t["status"]=="STABLE" and t["geometry_id"]==cache.geometry_id
            assert len(t["members"])==t["observation_count"]
            for member in t["members"]:
                row=cache.observations[member["observation_key"]]
                assert row["frame_id"]==member["frame_id"] and row["mask_local_id"]==member["mask_local_id"]
                members[member["frame_id"]].append((tid,member))
        for frame_number,fid in enumerate(sorted(members)):
            pack=cache.read_frame(fid,geometry_id=cache.geometry_id)
            sources.append(file_record(cache.root/cache.frames[fid]["support_file"]))
            for tid,member in members[fid]:
                at=int(np.searchsorted(pack["mask_local_ids"],member["mask_local_id"]))
                assert pack["mask_local_ids"][at]==member["mask_local_id"]
                lo,hi=pack["mask_offsets"][at:at+2]
                pts=pack["mask_surface_indices"][lo:hi]
                assert len(pts)==member["surface_support_count"]
                accum[tid][pts]=True
            if (frame_number+1)%100==0:print(json.dumps(dict(scene=scene,accumulated_frames=frame_number+1)),flush=True)
        full_support={tid:np.flatnonzero(mask).astype(np.int32) for tid,mask in accum.items()}
        del accum
        (output/"tracks"/scene).mkdir(parents=True)
        for tid,pts in full_support.items():
            path=output/"tracks"/scene/(tid+".npz")
            np.savez_compressed(path,point_indices=pts,geometry_id=cache.geometry_id,
                frame_ids=np.array([m["frame_id"] for m in tracks[tid]["members"]],np.int32))
            artifacts.append(file_record(path))
        packs={}
        for n,card in enumerate(cards):
            p=card["pair"]; key=p["pair_key"]; iid=p["target_instance_id"]
            base=scene+"_"+key[:16]
            paircolors=[palette[colors[t]] for t in p["track_ids"]]
            fs=figures[key]; assert len(fs)==2
            supports=[[],[]]; images=[]; events={e["frame_id"]:e for e in p["joint_observations"]}
            for f in fs:
                fid=f["frame_id"]; event=events[fid]
                assert f["track_ids"]==p["track_ids"] and f["mask_local_ids"]==event["mask_local_ids"]
                for rec in (f["rgb"], f["mask"]):verify_record(rec);sources.append(rec)
                if fid not in packs: packs[fid]=cache.read_frame(fid,geometry_id=cache.geometry_id)
                pack=packs[fid]
                with Image.open(f["rgb"]["path"]) as im:rgb=np.asarray(im.convert("RGB")).copy()
                with Image.open(f["mask"]["path"]) as im: labels=np.asarray(im).copy()
                overlay=rgb.astype(np.float32); masks=[]
                for j,mid in enumerate(f["mask_local_ids"]):
                    mask=labels==mid;assert mask.any();masks.append(mask)
                    at=int(np.searchsorted(pack["mask_local_ids"],mid))
                    assert pack["mask_local_ids"][at]==mid
                    lo,hi=pack["mask_offsets"][at:at+2]
                    pts=pack["mask_surface_indices"][lo:hi]
                    target_pts=pts[ids[pts]==iid]
                    assert len(target_pts)==event["target_support_points"][j]
                    assert len(pts)==event["mask_support_points"][j]
                    supports[j].append(target_pts)
                    color=np.asarray(paircolors[j],np.float32)
                    overlay[mask]=.55*overlay[mask]+.45*color
                    inner=mask.copy()
                    inner[1:] &=mask[:-1];inner[:-1]&=mask[1:]
                    inner[:,1:]&=mask[:,:-1];inner[:,:-1]&=mask[:,1:]
                    overlay[mask&~inner]=color
                assert not np.any(masks[0]&masks[1])
                box=f["crop_xyxy"]
                im=Image.fromarray(overlay.astype(np.uint8)).crop(box)
                im.thumbnail((1100,760))
                rel="cases/"+base+"_f"+str(fid)+".jpg";im.save(output/rel,quality=92)
                images.append(dict(src=rel,frame=fid,group=f["joint_view_group_id"],crop=box))
                artifacts.append(file_record(output/rel))
            two_frame_points=[len(np.unique(np.concatenate(v))) for v in supports]
            a,b=[full_support[tid] for tid in p["track_ids"]]
            original=np.flatnonzero(ids==iid)
            # Bit 0 = old instance, bit 1 = full A, bit 2 = full B.
            kinds=np.zeros(len(xyz),np.uint8);kinds[original]|=1;kinds[a]|=2;kinds[b]|=4
            shared=np.intersect1d(a,b,assume_unique=True)
            whole=np.flatnonzero(kinds>0)
            center=(xyz[whole].min(0)+xyz[whole].max(0))/2
            radius=max(float(np.linalg.norm(xyz[whole]-center,axis=1).max()),.05)
            nearby=np.all(np.abs(xyz-center)<radius+0.2,axis=1)&(kinds==0)
            context=sample(np.flatnonzero(nearby),6000)
            selected=[context]+[sample(np.flatnonzero(kinds==k),45000) for k in range(1,8)]
            indices=np.concatenate(selected).astype(np.int32)
            pose=np.asarray(cache.frames[fs[0]["frame_id"]]["camera_to_world"])
            points=(xyz[indices]-center)@pose[:3,:3]/radius
            cloud=np.column_stack((points,kinds[indices])).astype("<f4")
            rel="cases/"+base+".bin";cloud.tofile(output/rel);artifacts.append(file_record(output/rel))
            provenance="cases/"+base+"_indices.npz"
            np.savez_compressed(output/provenance,point_indices=indices,kinds=kinds[indices],geometry_id=cache.geometry_id)
            artifacts.append(file_record(output/provenance))
            results.append(dict(scene=scene,id=iid,key=key,tracks=p["track_ids"],
                frames=p["joint_frame_count"],views=p["independent_joint_view_groups"],
                min_fraction=p["minimum_joint_target_fraction"],colors=paircolors,
                cloud=rel,point_count=len(indices),original_points=len(original),
                displayed_original_points=int(np.count_nonzero(kinds[indices]&1)),
                full_support_points=[len(a),len(b)],
                intersection_points=[int(np.count_nonzero(ids[v]==iid)) for v in (a,b)],
                previous_two_frame_intersection_points=two_frame_points,
                member_counts=[tracks[tid]["observation_count"] for tid in p["track_ids"]],
                support_definition="union of all assigned members of each stable track; no old-ID restriction",
                shared_points=len(shared),support_points=[len(a),len(b)],images=images,
                roi_outside=p["roi_comparison"]["outside_all_selected_roi_context"],
                evidence_frames=[f["frame_id"] for f in fs],geometry_id=cache.geometry_id))
            if (n+1)%10==0:print(json.dumps(dict(scene=scene,cases=n+1)),flush=True)
        print(json.dumps(dict(scene=scene,cases=len(cards),status="PASS")),flush=True)
    results.sort(key=lambda c:(c["scene"]!="room0",
        0 if (c["scene"]=="room0" and c["id"]==162 and c["roi_outside"]) or (c["scene"]=="office2" and c["id"]==94 and c["roi_outside"]) else 1,
        -c["min_fraction"],c["id"],c["key"]))
    write_json(output/"visual_data.json",dict(dataset="Replica",cloud_schema="membership_bits_v2",cases=results,original_results_modified=False))
    template=Path(__file__).with_name("two_audit_visual.html")
    shutil.copyfile(template,output/"index.html")
    # Retain the detailed numerical page as an optional reference.
    (output/"legacy").symlink_to(root/"inspection",target_is_directory=True)
    for r in {r["path"]:r for r in sources}.values():verify_record(r)
    report=dict(status="PASS",cases=len(results),images=sum(len(c["images"]) for c in results),
        source_records=list({r["path"]:r for r in sources}.values()),artifacts=artifacts,
        geometry_checked=True,mask_support_counts_checked=True,full_stable_track_union=True,colors_match_2d_3d=True,
        original_results_modified=False,selection="same 87 representative pairs as original review; reordered for display",
        builder=file_record(Path(__file__)),template=file_record(template),
        data=file_record(output/"visual_data.json"),index=file_record(output/"index.html"))
    write_json(output/"validation.json",report)
    print(json.dumps(dict(status="PASS",cases=len(results),images=report["images"])),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();build(a.root,a.output)
