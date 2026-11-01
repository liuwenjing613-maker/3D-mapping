"""Same full-pool ABC evidence as the reviewed experiment, with per-run inputs."""
from pathlib import Path
from itertools import combinations
from functools import lru_cache
from collections import Counter
import argparse
import hashlib
import copy
import json
import numpy as np
from PIL import Image

from .seed_final_abc import read, sha, save, require, passes, load_context
from .seed_abc_projection import source_cloud, cloud_visibility, correspondence, rank_views


def check_pair(a, b, c):
    require(a.shape == b.shape == c.shape, "Projected shapes differ")
    union = b | c
    areas = dict(A=int(a.sum()), B=int(b.sum()), C=int(c.sum()),
                 union_BC=int(union.sum()), covered_A=int((a & union).sum()),
                 intersection_BC=int((b & c).sum()))
    valid = min(areas[k] for k in ("A", "B", "C")) > 0
    coverage = areas["covered_A"] / areas["A"] if areas["A"] else 0.
    overlap = areas["intersection_BC"] / min(areas["B"], areas["C"]) if valid else 1.
    return dict(areas=areas, valid=valid, coverage=coverage, overlap=overlap,
                passed=valid and 10 * areas["covered_A"] > 7 * areas["A"]
                and 10 * areas["intersection_BC"] < min(areas["B"], areas["C"]))


class Inputs:
    def __init__(self, context, dest):
        self.context = context
        self.dest = Path(dest)
        require(not self.dest.exists(), "Evidence output must be new; stale projections are not reused")
        self.dest.mkdir(parents=True)
        self.locks = {}
        self.v = Path(context["mask_root"])
        self.g = self.load(context["grouping"])
        self.scene = self.g["rule"].get("scene", "unspecified")
        self.seeds = {s["id"]: s for s in self.g["seeds"]}
        require(len(self.seeds) == len(self.g["seeds"]), "Repeated candidate ID")
        require([s["index"] for s in self.g["seeds"]] == list(range(len(self.seeds))),
                "Candidate/support order differs")
        self.primary = {self.g["seeds"][g["primary"]]["id"]: g["id"] for g in self.g["groups"]}
        self.cfg = self.load(context["mapping_config"])
        self.sr = Path(self.cfg["source"]["scene_root"])
        self.geom = self.load(context["geometry"])
        self.frames = self.geom["frame_ids"]
        require(self.geom["no_future_geometry"] and len(self.frames) == self.geom["frame_count"]
                and self.frames == sorted(set(self.frames)) and bool(self.frames),
                "Require a unique ordered arrived-frame prefix")
        self.actual = set()
        self.preflight_excluded = set()
        self.cam = self.cfg["camera"]
        self.shape = (int(self.cam["height"]), int(self.cam["width"]))
        traj = self.sr / self.cfg["source"].get("trajectory", "traj.txt")
        self.lock(traj)
        self.poses = np.loadtxt(traj).reshape(-1, 4, 4)
        require(min(self.frames) >= 0 and max(self.frames) < len(self.poses), "Unknown arrived pose")
        require(np.isfinite(self.poses[self.frames]).all(), "Invalid arrived pose")
        self.lock(context["supports"])
        self.lock(context["pairs"])
        with np.load(context["supports"], allow_pickle=False) as z:
            self.indices, self.offsets = z["indices"].copy(), z["offsets"].copy()
        require(len(self.offsets) == len(self.seeds) + 1 and self.offsets[0] == 0
                and self.offsets[-1] == len(self.indices)
                and np.all(np.diff(self.offsets) >= 0), "Invalid support offsets")
        self.support = {s["id"]: self.indices[self.offsets[s["index"]]:self.offsets[s["index"]+1]].copy()
                        for s in self.g["seeds"]}
        with np.load(context["pairs"], allow_pickle=False) as z:
            self.inter = z["intersection"].copy()
        require(self.inter.shape == (len(self.seeds), len(self.seeds)), "Wrong pair dimensions")
        for mid, seed in self.seeds.items():
            require(seed["noise_decision"]["exclude"] is False and seed["frame"] in self.frames,
                    "Ineligible or future candidate")
            support = self.support[mid]
            require(len(support) == seed["support_count"] and
                    hashlib.sha256(support.tobytes()).hexdigest() == seed["support_sha256"],
                    "Support identity mismatch")
            require(np.all(support >= 0) and len(np.unique(support)) == len(support),
                    "Invalid surface support")
            self.mask(mid)  # Bind every legal mask, including evidence-only candidates.

    def lock(self, path):
        path = Path(path).resolve()
        value = sha(path)
        require(str(path) not in self.locks or self.locks[str(path)] == value, "Input changed")
        self.locks[str(path)] = value
        return value

    def load(self, path):
        self.lock(path)
        return read(path)

    @lru_cache(maxsize=32)
    def mask(self, mid):
        seed = self.seeds[mid]
        require(self.lock(self.v / seed["mask"]) == seed["mask_sha256"], "Mask file changed")
        raw = np.asarray(Image.open(self.v / seed["mask"]))
        require(raw.ndim == 2 and np.isin(raw, [0, 1, 255]).all(), "Mask must be binary")
        mask = raw > 0
        require(mask.shape == self.shape and int(mask.sum()) == seed["area"], "Mask geometry changed")
        require(hashlib.sha256(seed["frame"].to_bytes(4, "little") +
                               np.packbits(mask.ravel()).tobytes()).hexdigest() == seed["digest"],
                "Mask/frame digest changed")
        return mask

    @lru_cache(maxsize=128)
    def visible(self, frame):
        require(frame in self.frames, "Future visibility frame")
        path = Path(self.context["visible_root"]) / ("frame%06d_visible.npz" % frame)
        self.lock(path)
        with np.load(path, allow_pickle=False) as z:
            return z["vertex_index"].copy(), z["uv"].copy()

    def check_sources(self):
        for path, expected in self.locks.items():
            require(sha(path) == expected, "Input changed: " + path)

class Search(Inputs):
 def __init__(self,context,dest):
  super().__init__(context,dest)
  prepared=self.load(context['semantics']);self.semantics=prepared['records']
  require(set(prepared['records'])==set(self.seeds), 'Semantics must cover the whole legal pool')
  require(prepared['source_hashes'].get(str(Path(context['grouping']).resolve()))==sha(context['grouping']), 'Semantics belong to different grouping')
  for p,h in prepared['source_hashes'].items():require(sha(p)==h, 'Stale semantic source')
  self.locks.update(prepared['source_hashes'])
  self.eligible=[m for m,s in self.seeds.items() if not self.semantics[m]['exclude']]
  self.candidates=sorted(self.eligible,key=lambda m:(m not in self.primary,-self.seeds[m]['support_count'],self.seeds[m]['frame'],self.seeds[m]['representative']['model'],self.seeds[m]['digest']))
  self.index={m:s['index'] for m,s in self.seeds.items()}
  self.pose_scores={};self.visibility={};self.views_tested=0;self.projection_count=0
  self.membership={}
  self.frames=self.geom['frame_ids'];self.cam=self.cfg['camera']
  for p in [Path(__file__),Path(__file__).with_name('seed_abc_projection.py')]:self.lock(p)
 def inter_count(self,a,b):return int(self.inter[self.index[a],self.index[b]])
 def prepare_relations(self):
  path=self.dest/'relations.npz'
  mids=list(self.seeds);frames=sorted({s['frame'] for s in self.seeds.values()});frame_index={f:i for i,f in enumerate(frames)}
  counts=np.zeros((len(mids),len(frames)),np.int32)
  n=int(self.indices.max())+1 if len(self.indices) else 0
  for j,f in enumerate(frames):
   vi,_=self.visible(f);flag=np.zeros(max(n,int(vi.max())+1 if len(vi) else 0),bool);flag[vi]=True
   for i,mid in enumerate(mids):counts[i,j]=int(flag[self.support[mid]].sum())
  self.common_counts=counts;self.frame_index=frame_index
  np.savez_compressed(path,mask_ids=np.asarray(mids),frames=np.asarray(frames),counts=counts)
  def duplicate(a,b):
   i=self.inter_count(a,b);na=int(counts[self.index[a],frame_index[self.seeds[b]['frame']]]);nb=int(counts[self.index[b],frame_index[self.seeds[a]['frame']]])
   require(i<=min(na,nb), 'Invalid covisible intersection')
   return na>0 and nb>0 and 10*i>9*na and 10*i>9*nb
  remaining=self.candidates[:];classes=[]
  while remaining:
   p=remaining[0];members=[p]+[m for m in remaining[1:] if duplicate(p,m)]
   classes.append(dict(representative=p,members=members))
   taken=set(members);remaining=[m for m in remaining if m not in taken]
  self.classes=classes;self.reps=[c['representative'] for c in classes]
  self.class_by_rep={c['representative']:c['members'] for c in classes}
  self.rep_of={m:c['representative'] for c in classes for m in c['members']}
  plans=[]
  for a in self.reps:
   ns=[b for b in self.reps if b!=a and len(self.support[b])>=20 and self.inter_count(a,b)>=20 and 5*self.inter_count(a,b)>=min(len(self.support[a]),len(self.support[b]))]
   pairs=list(combinations(ns,2))
   pairs.sort(key=lambda bc:(self.inter_count(*bc)/min(len(self.support[bc[0]]),len(self.support[bc[1]])),-(self.inter_count(a,bc[0])+self.inter_count(a,bc[1])),bc))
   plans.append(dict(A=a,neighbors=ns,pairs=pairs))
  save(self.dest/'search_plan.json',dict(legal=len(self.seeds),semantic_survivors=len(self.eligible),representatives=len(self.reps),classes=classes,plans=plans,total_possible_pairs=sum(len(p['pairs']) for p in plans),original_primary_masks=list(self.primary),old_final_masks=sorted(self.actual),unchanged_known_projection_excluded=sorted(self.preflight_excluded)))
  print('PLAN',self.scene,'legal',len(self.seeds),'sem',len(self.eligible),'reps',len(self.reps),'pairs',sum(len(p['pairs']) for p in plans),flush=True)
  return plans
 @lru_cache(maxsize=400)
 def depth(self,frame):
  require(frame in self.frames, 'Future depth frame')
  p=self.sr/self.cfg['source']['depth_pattern'].format(frame=frame);self.lock(p)
  return np.asarray(Image.open(p),np.float64)/self.cam['depth_png_units_per_meter']
 @lru_cache(maxsize=128)
 def cloud(self,mid):
  s=self.seeds[mid];return source_cloud(self.mask(mid),self.depth(s['frame']),self.poses[s['frame']],self.cam)
 def fraction(self,mid,target):
  key=(mid,target)
  if key not in self.visibility:
   self.visibility[key]=1. if self.seeds[mid]['frame']==target else cloud_visibility(self.cloud(mid),self.depth(target),self.poses[target],self.cam)
   self.views_tested+=1
  return self.visibility[key]
 def ranking(self,ids,anchor):
  src=sorted({self.seeds[m]['frame'] for m in ids})
  for sf in src:
   if sf not in self.pose_scores:
    scores={}
    S=self.poses[sf]
    for f in self.frames:
     T=self.poses[f];t=float(np.linalg.norm(T[:3,3]-S[:3,3]))
     ang=float(np.degrees(np.arccos(np.clip((np.trace(T[:3,:3].T@S[:3,:3])-1)/2,-1,1))))
     scores[f]=t/.1+ang/10
    self.pose_scores[sf]=scores
  pf=self.seeds[anchor]['frame'];rank=[]
  for f in self.frames:
   scores=[self.pose_scores[s][f] for s in src];rank.append((round(max(scores),10),round(float(np.mean(scores)),10),f!=pf,f))
  rank.sort()
  return [x[3] for x in rank]
 @lru_cache(maxsize=8)
 def correspondence(self,source,target):
  return correspondence(self.depth(target),self.depth(source),self.poses[target],self.poses[source],self.cam)
 @lru_cache(maxsize=128)
 def projected(self,mid,target):
  sf=self.seeds[mid]['frame'];p=self.dest/'projections'/f'{mid}_f{target:06d}.npz'
  if p.exists():
   with np.load(p) as z:
    assert str(z['source_sha256'])==self.seeds[mid]['mask_sha256']
    return np.unpackbits(z['packed'],count=int(np.prod(self.shape))).reshape(self.shape).astype(bool)
  mask=self.mask(mid)
  if sf==target:proj=mask.copy()
  else:
   ix,known=self.correspondence(sf,target);proj=mask.ravel()[ix]&known
  p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,packed=np.packbits(proj.ravel()),source_sha256=self.seeds[mid]['mask_sha256'],source_frame=sf,target_frame=target)
  self.projection_count+=1
  return proj
 def triple(self,a,b,c):
  ids=[a,b,c];key=a+'__'+'__'.join(sorted([b,c]));p=self.dest/'triples'/f'{key}.json'
  if p.exists():
   rec=read(p);assert rec['ids']==ids and all(rec['mask_hashes'][m]==self.seeds[m]['mask_sha256'] for m in ids);return rec
  ranking=self.ranking(ids,a)
  if not hasattr(self,'rank_checks'):self.rank_checks=0
  if self.rank_checks<4:
   expected=rank_views(self.poses,self.frames,[self.seeds[m]['frame'] for m in ids],self.seeds[a]['frame'])
   assert ranking==[r['frame'] for r in expected];self.rank_checks+=1
  chosen=best=None;tried=0
  for f in ranking:
   fs={m:self.fraction(m,f) for m in ids};v=dict(frame=f,min_visible=min(fs.values()),visible_fractions=fs);tried+=1
   if best is None or v['min_visible']>best['min_visible']:best=v
   if v['min_visible']>=.95:chosen=v;break
  qualified=chosen is not None;chosen=chosen or best;tf=chosen['frame']
  proj={m:self.projected(m,tf) for m in ids};metric=check_pair(proj[a],proj[b],proj[c])
  is_largest=int(proj[a].sum())==max(int(x.sum()) for x in proj.values())
  rec=dict(ids=ids,A=a,B=b,C=c,target_frame=tf,view_qualified=qualified,view_choice=chosen,views_searched=tried,is_large_A=is_largest,metric=metric,passed=is_largest and metric['passed'],mask_hashes={m:self.seeds[m]['mask_sha256'] for m in ids})
  save(p,rec);return rec
 def run(self):
  self.prepare_relations()
  plan=self.load(self.dest/'search_plan.json');self.classes=plan['classes'];self.class_by_rep={c['representative']:c['members'] for c in self.classes}
  records=[]
  for no,p in enumerate(plan['plans'],1):
   a=p['A'];evaluations=[]
   for b,c in p['pairs']:
    r=copy.deepcopy(self.triple(a,b,c));m=r['metric'];areas=m.get('areas',{});union=areas.get('union_BC',0);covered=areas.get('covered_A',0)
    reverse=covered/union if union else 0.
    checks=dict(original_geometric=bool(r['passed']),visibility=r['view_choice']['min_visible']>=.8,reverse_coverage=union>0 and 10*covered>7*union)
    require(all(checks.values())==passes(r), 'ABC metric disagreement')
    r['revised']=dict(passed=all(checks.values()),checks=checks,reverse_coverage=reverse,balanced_coverage=min(m.get('coverage',0),reverse),failed=[k for k,v in checks.items() if not v])
    evaluations.append(r)
   hits=[r for r in evaluations if r['revised']['passed']]
   hits.sort(key=lambda r:(-r['revised']['balanced_coverage'],-r['view_choice']['min_visible'],r['metric']['overlap'],-r['metric']['coverage'],r['B'],r['C']))
   rec=dict(A=a,status='SELECTED' if hits else 'NO_RELIABLE_MATCH' if evaluations else 'NEED_THIRD',accepted=bool(hits),neighbors=p['neighbors'],possible_pairs=len(p['pairs']),tested_pairs=len(evaluations),exhaustive=True,passing_count=len(hits),hits=hits,chosen=hits[0] if hits else None,evaluations=evaluations,duplicate_class=self.class_by_rep[a],historical_geometric_passes=sum(r['passed'] for r in evaluations))
   save(self.dest/'anchors'/f'{a}.json',rec);records.append(rec)
   save(self.dest/'progress.json',dict(scene=self.scene,done=no,total=len(plan['plans']),statuses=dict(Counter(x['status'] for x in records)),tested_pairs=sum(x['tested_pairs'] for x in records)))
   print('ANCHOR',self.scene,no,'/',len(plan['plans']),a,rec['status'],len(hits),'/',len(evaluations),flush=True)
  self.check_sources()
  save(self.dest/'search.json',dict(status='PASS',anchors=records,classes=self.classes,legal_masks=len(self.seeds),semantic_survivors=len(self.eligible),representatives=len(plan['plans']),statuses=dict(Counter(x['status'] for x in records)),tested_pairs=sum(x['tested_pairs'] for x in records),possible_pairs=plan['total_possible_pairs'],policy=dict(min_visibility=.8,coverage_strict=.7,reverse_coverage_strict=.7,overlap_strict=.1,selection='max min(forward coverage, reverse coverage), then max visibility, min overlap, max forward coverage, B/C IDs',exhaustive=True)))
  save(self.dest/'source_lock.json',self.locks)
  print('SEARCH_DONE',self.scene,dict(Counter(x['status'] for x in records)),flush=True)
  return read(self.dest/'search.json')
