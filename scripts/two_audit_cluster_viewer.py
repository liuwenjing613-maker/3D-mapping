"""Read-only complete two-audit cluster viewer; no clustering or map edits."""
import argparse, io, json, os, shutil, threading, time
from collections import defaultdict, Counter
from functools import lru_cache, partial
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image
from instance_repair.two_audit_cache import ObservationCache, write_json
from instance_repair.two_audit_sources import file_record, verify_record

SCENES = ("room0", "office2")
def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def sample(a,n):
    return a[np.linspace(0,len(a)-1,n,dtype=int)] if n and len(a)>n else a
def support(pack,mid):
    j=int(np.searchsorted(pack["mask_local_ids"],mid))
    assert j<len(pack["mask_local_ids"]) and pack["mask_local_ids"][j]==mid
    lo,hi=pack["mask_offsets"][j:j+2]
    return pack["mask_surface_indices"][lo:hi]
def build(root,out):
    root=Path(root);out=Path(out)
    out.mkdir(exist_ok=False)
    catalog=[]; private={}; reports={}; source_records=[]
    old=json.loads((root/"visual_review_full_v2/visual_data.json").read_text())
    oldcolors={(c["scene"],t):color for c in old["cases"] for t,color in zip(c["tracks"],c["colors"])}
    for scene in SCENES:
        tr=root/"stable_tracks"/scene
        manifest=json.loads((tr/"manifest.json").read_text())
        for name in ("tracks.json","pending_observations.jsonl"):
            rec=file_record(tr/name);assert rec["sha256"]==manifest["artifact_sha256"][name];source_records.append(rec)
        cache=ObservationCache(manifest["cache_root"])
        assert cache.geometry_id==manifest["geometry_id"] and cache.manifest["cache_id"]==manifest["cache_id"]
        tracks=json.loads((tr/"tracks.json").read_text());pending=rows(tr/"pending_observations.jsonl")
        native=Path(cache.manifest["mapping_root"])/"baseline/native/instance_surface.npz"
        nr=file_record(native)
        lock=[x for x in cache.manifest["locked_sources"] if x["resolved_path"]==str(native.resolve())]
        assert len(lock)==1 and lock[0]["sha256"]==nr["sha256"]
        source_records.append(nr)
        with np.load(native,allow_pickle=False) as z: xyz=z["xyz_m"]
        cache.require_geometry(xyz)
        cfg=json.loads((Path(cache.manifest["mapping_root"])/"config.json").read_text())["source"]
        metadata={x["frame"]:x for x in rows(Path(cfg["mask_root"])/cfg["mask_metadata"]) if x["frame"] in cache.frames}
        frames={}
        for fid,frame in cache.frames.items():
            rgb=Path(cfg["scene_root"])/cfg["rgb_pattern"].format(frame=fid)
            rec=file_record(rgb);verify_record(frame["sources"]["mask"])
            assert rec["sha256"]==metadata[fid]["input_sha256"]
            assert frame["sources"]["mask"]["sha256"]==metadata[fid]["mask_sha256"]
            frames[fid]=dict(rgb=str(rgb),rgb_sha256=rec["sha256"],mask=frame["sources"]["mask"])
        members_by_frame=defaultdict(list); assigned=set()
        for t in tracks:
            assert len(t["members"])==t["observation_count"] and t["geometry_id"]==cache.geometry_id
            assert len({m["frame_id"] for m in t["members"]})==len(t["members"])
            for m in t["members"]:
                key=m["observation_key"];assert key not in assigned;assigned.add(key)
                row=cache.observations[key]
                assert all(row[k]==m[k] for k in ("frame_id","mask_local_id","surface_support_count"))
                members_by_frame[m["frame_id"]].append((t["track_id"],m))
        pendingkeys={p["observation_key"] for p in pending}
        assert len(pendingkeys)==len(pending) and not assigned&pendingkeys
        assert assigned|pendingkeys==set(cache.observations)
        accum=defaultdict(list)
        for n,fid in enumerate(cache.frames):
            pack=cache.read_frame(fid,geometry_id=cache.geometry_id)
            for tid,m in members_by_frame[fid]:
                pts=support(pack,m["mask_local_id"]).astype(np.int32)
                assert len(pts)==m["surface_support_count"]
                accum[tid].append(pts)
            if (n+1)%100==0:print(json.dumps(dict(scene=scene,frames=n+1)),flush=True)
        (out/scene).mkdir()
        public=[]
        for t in tracks:
            tid=t["track_id"]
            pts,hits=np.unique(np.concatenate(accum.pop(tid)),return_counts=True)
            assert int(hits.sum())==sum(m["surface_support_count"] for m in t["members"])
            np.savez_compressed(out/scene/(tid+".npz"),indices=pts,hits=hits.astype(np.uint16),geometry_id=cache.geometry_id)
            center=(xyz[pts].min(0)+xyz[pts].max(0))/2
            radius=max(float(np.linalg.norm(xyz[pts]-center,axis=1).max()),.05)
            view_member=max(t["members"],key=lambda m:m["surface_support_count"])
            rot=np.asarray(cache.frames[view_member["frame_id"]]["camera_to_world"])[:3,:3]
            color=oldcolors.get((scene,tid))
            if color is None:
                import colorsys
                color=[int(v*255) for v in colorsys.hsv_to_rgb((int(tid[1:])*.61803398875)%1,.72,.96)]
            detail=dict(scene=scene,id=tid,status=t["status"],members=t["members"],points=len(pts),
                        center=center.tolist(),radius=radius,rotation=rot.tolist(),color=color,
                        views=t["independent_view_groups"],min_iou=t["minimum_admitted_covisible_iou"])
            write_json(out/scene/(tid+".json"),detail)
            entry={k:detail[k] for k in ("scene","id","status","points","color","views")}
            entry.update(frames=len(t["members"]),first=t["representative_frame_id"],mask=t["representative_mask_local_id"])
            catalog.append(entry);public.append(entry)
        for p in pending:
            row=cache.observations[p["observation_key"]]
            tid="U%06d_%03d"%(p["frame_id"],p["mask_local_id"])
            member=dict(observation_key=p["observation_key"],frame_id=p["frame_id"],mask_local_id=p["mask_local_id"],
                        surface_support_count=row["surface_support_count"],view_group_id=1,fixed_representative_evidence=None)
            detail=dict(scene=scene,id=tid,status=p["status"],members=[member],points=row["surface_support_count"],
                        color=[170,180,190],views=1,min_iou=None,matched_tracks=p["matched_track_ids"])
            write_json(out/scene/(tid+".json"),detail)
            catalog.append(dict(scene=scene,id=tid,status=p["status"],points=detail["points"],
                                color=detail["color"],views=1,frames=1,first=p["frame_id"],mask=p["mask_local_id"]))
        reports[scene]=dict(tracks=len(tracks),stable=sum(t["status"]=="STABLE" for t in tracks),
            provisional=sum(t["status"]=="PROVISIONAL" for t in tracks),assigned=len(assigned),
            pending=len(pending),observations=len(cache.observations),geometry_id=cache.geometry_id)
        assert len(tracks)==manifest["totals"]["tracks"] and len(assigned)==manifest["totals"]["assigned_observations"]
        private[scene]=dict(cache_root=str(cache.root),native=str(native),frames=frames)
        print(json.dumps(dict(scene=scene,status="PASS",**reports[scene])),flush=True)
    for rec in source_records:verify_record(rec)
    write_json(out/"catalog.json",dict(scenes=reports,items=catalog,kind="two-audit observation clusters; not final instance map"))
    write_json(out/"sources.json",private)
    shutil.copyfile(Path(__file__).with_name("two_audit_clusters.html"),out/"index.html")
    write_json(out/"validation.json",dict(status="PASS",scenes=reports,total_tracks=sum(x["tracks"] for x in reports.values()),
        all_observations_accounted_for=True,all_frame_packs_verified=True,all_rgb_mask_hashes_verified=True,
        source_records=source_records,algorithm_modified=False,map_modified=False))
    print("BUILD PASS",flush=True)

class Viewer:
    def __init__(self,out):
        self.out=Path(out); self.sources=json.loads((self.out/"sources.json").read_text())
        self.xyz={};self.caches={};self.lock=threading.RLock()
        self.catalog={(x["scene"],x["id"]):x for x in json.loads((self.out/"catalog.json").read_text())["items"]}
        for s,rec in self.sources.items():
            with np.load(rec["native"],allow_pickle=False) as z:self.xyz[s]=z["xyz_m"]
            self.caches[s]=ObservationCache(rec["cache_root"])
            self.caches[s].require_geometry(self.xyz[s])
    @lru_cache(maxsize=64)
    def detail(self,s,t):
        assert (s,t) in self.catalog
        return json.loads((self.out/s/(t+".json")).read_text())
    @lru_cache(maxsize=3)
    def pack(self,s,fid):
        c=self.caches[s];return c.read_frame(fid,geometry_id=c.geometry_id)
    @lru_cache(maxsize=4)
    def images(self,s,fid):
        f=self.sources[s]["frames"][str(fid)]
        verify_record(f["mask"])
        assert file_record(f["rgb"])["sha256"]==f["rgb_sha256"]
        with Image.open(f["rgb"]) as im:rgb=np.asarray(im.convert("RGB")).copy()
        with Image.open(f["mask"]["path"]) as im:labels=np.asarray(im).copy()
        assert rgb.shape[:2]==labels.shape
        return rgb,labels
    @lru_cache(maxsize=256)
    def image(self,s,t,j,crop,overlay,thumb):
        d=self.detail(s,t);m=d["members"][j]
        with self.lock:rgb,labels=self.images(s,m["frame_id"])
        mask=labels==m["mask_local_id"];assert mask.any()
        im=rgb.astype(np.float32)
        if overlay:
            color=np.array(d["color"]);im[mask]=.58*im[mask]+.42*color
            inner=mask.copy();inner[1:]&=mask[:-1];inner[:-1]&=mask[1:];inner[:,1:]&=mask[:,:-1];inner[:,:-1]&=mask[:,1:]
            im[mask&~inner]=color
        im=Image.fromarray(im.astype(np.uint8))
        if crop:
            yy,xx=np.where(mask);margin=max(35,int(max(xx.ptp(),yy.ptp())*.15))
            im=im.crop((max(0,int(xx.min())-margin),max(0,int(yy.min())-margin),min(im.width,int(xx.max())+margin+1),min(im.height,int(yy.max())+margin+1)))
        im.thumbnail((180,110) if thumb else (1200,800))
        b=io.BytesIO();im.save(b,format="JPEG",quality=85 if thumb else 92)
        return b.getvalue()
    @lru_cache(maxsize=10)
    def union(self,s,t):
        d=self.detail(s,t)
        if t.startswith("U"):
            m=d["members"][0]
            with self.lock:pts=support(self.pack(s,m["frame_id"]),m["mask_local_id"]).copy()
            hits=np.ones(len(pts),np.uint16)
        else:
            with np.load(self.out/s/(t+".npz"),allow_pickle=False) as z:pts=z["indices"];hits=z["hits"]
        xyz=self.xyz[s][pts]
        if len(pts):
            center=(xyz.min(0)+xyz.max(0))/2;radius=max(float(np.linalg.norm(xyz-center,axis=1).max()),.05)
        else:center=np.zeros(3);radius=1.
        view_member=max(d["members"],key=lambda m:m["surface_support_count"])
        rot=np.asarray(self.caches[s].frames[view_member["frame_id"]]["camera_to_world"])[:3,:3]
        return pts,hits,center,radius,rot
    def cloud(self,s,t,j,kind,limit):
        pts,hits,center,radius,rot=self.union(s,t);d=self.detail(s,t)
        if kind=="member":
            m=d["members"][j]
            with self.lock:pts=support(self.pack(s,m["frame_id"]),m["mask_local_id"]).copy()
            hits=np.ones(len(pts),np.uint16)
        elif kind=="context":
            pts=sample(np.arange(len(self.xyz[s]),dtype=np.int32),40000);hits=np.zeros(len(pts))
        ix=sample(np.arange(len(pts)),limit)
        return np.column_stack(((self.xyz[s][pts[ix]]-center)@rot/radius,hits[ix])).astype("<f4").tobytes()

def serve(root,out,state,port):
    viewer=Viewer(out)
    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff");super().end_headers()
        def list_directory(self,path):
            self.send_error(403);return None
        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path.startswith("/cluster-api/"):
                try:
                    q=parse_qs(parsed.query);s=q["scene"][0];t=q["id"][0]
                    assert (s,t) in viewer.catalog
                    j=int(q.get("member",["0"])[0]);d=viewer.detail(s,t)
                    assert 0<=j<len(d["members"])
                    if parsed.path=="/cluster-api/image":
                        data=viewer.image(s,t,j,q.get("crop",["1"])[0]=="1",q.get("overlay",["1"])[0]=="1",q.get("thumb",["0"])[0]=="1")
                        content="image/jpeg"
                    elif parsed.path=="/cluster-api/cloud":
                        kind=q.get("kind",["union"])[0];assert kind in ("union","member","context")
                        limit=int(q.get("limit",["150000"])[0]);assert limit in (0,150000)
                        data=viewer.cloud(s,t,j,kind,limit);content="application/octet-stream"
                    else:self.send_error(404);return
                    self.send_response(200);self.send_header("Content-Type",content);self.send_header("Content-Length",str(len(data)));self.end_headers();self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):pass
                except Exception as e:
                    print("API ERROR",repr(e),flush=True);self.send_error(500,"Source validation or request failed")
                return
            if parsed.path.endswith("/sources.json") or parsed.path.endswith(".npz"):
                self.send_error(403);return
            super().do_GET()
    server=ThreadingHTTPServer(("127.0.0.1",port),partial(Handler,directory=root))
    write_json(state,dict(status="RUNNING",bind="127.0.0.1",port=server.server_port,pid=os.getpid(),root=root,cluster_root=out))
    print("READY",flush=True);server.serve_forever()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["build","serve"]);p.add_argument("--root",required=True);p.add_argument("--output",required=True);p.add_argument("--state");p.add_argument("--port",type=int,default=42449)
    a=p.parse_args()
    if a.action=="build":build(a.root,a.output)
    else:serve(a.root,a.output,a.state,a.port)
