"""Correct reduced+1 GT names and add independent GT instance references."""
from pathlib import Path
import json,zipfile
import numpy as np
from scipy.spatial import cKDTree
from detect_rois import sha
from analyse_conflicts import classname
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_room0_roi_review_20261002')
OUT=ROOT/'conflict_diagnosis'
cases=json.loads((OUT/'conflict_cases.json').read_text())
with np.load('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/ground_truth/gt.npz') as z:
 xyz=z['xyz_ref']; sem=z['semantic_id']; gi=z['instance_id']; meta=json.loads(str(z['metadata_json']))
manifest=json.loads(Path(meta['source_manifest']).read_text());classes=manifest['semantic_classes']
assert 'reduced + 1' in manifest['ground_truth']
src=Path('/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/final/room0/P1-A1/diffusion/holes_geodesic/instance_surface.npz')
with np.load(src) as z:predxyz=z['xyz_m'];predid=z['instance_id']
tree=cKDTree(predxyz[predid>0]);assigned=np.flatnonzero(predid>0)
dist,idx=tree.query(xyz,workers=8);mapped=predid[assigned[idx]];good=(dist<=.01)&(sem>0)
detail={}
all_ids={d['id'] for r in cases for d in r['instance_details']}
for i in all_ids:
 ids=np.flatnonzero(good&(mapped==i));v,c=np.unique(sem[ids],return_counts=True);order=np.argsort(-c)
 names=[{'semantic_id':int(v[j]),'name':classname(classes,int(v[j])),'GT_vertices':int(c[j]),'fraction':float(c[j]/max(1,len(ids)))} for j in order[:3]]
 v,c=np.unique(gi[ids][gi[ids]>0],return_counts=True);order=np.argsort(-c)
 objects=[{'GT_instance_id':int(v[j]),'GT_vertices':int(c[j]),'fraction_of_nonzero_GT_instances':float(c[j]/max(1,c.sum()))} for j in order[:3]]
 detail[i]=(names,objects)
gttree=cKDTree(xyz)
for r in cases:
 for d in r['instance_details']:d['semantic_reference'],d['GT_instance_reference']=detail[d['id']]
 for p in r['competition_pairs']:
  dis,ix=gttree.query(p['sample_xyz_m']);p['sample_GT_reference']={'distance_m':float(dis),'GT_instance_id':int(gi[ix]),'semantic_id':int(sem[ix]),'name':classname(classes,int(sem[ix])),'within_1cm':bool(dis<=.01)}
(OUT/'conflict_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
audit=json.loads((OUT/'diagnosis_audit.json').read_text());audit['GT_semantic_encoding']='official Replica_map_to_reduced + 1; 0 unlabeled';audit['posthoc_GT_instance_references_added']=True
assert sha(src)==audit['source_hashes']['final']
(OUT/'diagnosis_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{'roi':r['roi_id'],'IDs':[{'id':d['id'],'name':d['semantic_reference'][:1],'GT':d['GT_instance_reference'][:1]} for d in r['instance_details']]} for r in cases],ensure_ascii=False))
