from pathlib import Path
import sys,json,hashlib,time,collections,zipfile,itertools
import numpy as np
import cv2
from PIL import Image
from selection_rules import choose, noise_reason
R=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/B/candidates/versions');R.mkdir(parents=True,exist_ok=True)
F=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/B/candidates/overmerge')
I=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/continuation300/200/B/candidates/intersection/web/intersection.json')
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    start=time.time();f=read(F/'web/data.json');ir=read(I);hashes=read(F/'input_hashes.json')
    protected={str(p):sha(p) for p in [I,F/'web/data.json',F/'input_hashes.json']};cache=collections.OrderedDict();verified=set()
    def pixels(sid):
        if sid in cache:cache.move_to_end(sid);return cache[sid]
        src=f['sources'][sid];path=Path(src['path'])
        if str(path) not in verified:assert sha(path)==hashes[str(path)];verified.add(str(path))
        labels=np.asarray(Image.open(path),np.int32);x0,y0,x1,y1=src['box'];assert labels.shape==(y1-y0,x1-x0)
        areas=np.bincount(labels.ravel());full=np.zeros((680,1200),np.int32);full[y0:y1,x0:x1]=labels
        for lid,cid in src['mapping'].items():assert int(areas[int(lid)])==f['candidates'][cid]['pixels']
        cache[sid]=(full,areas)
        if len(cache)>18:cache.popitem(last=False)
        return full,areas
    out=dict(schema='mask-version-selection-v2',rule=dict(group_min_smaller_coverage=.5,near_iou=.9,near_requires_mutual_best=True,containment=.95,comparison='full 2D masks in original frame coordinates',single_remaining='select directly per user; flag crop/reference risks',two_cropformers_agree_without_sam='select original per user',reference_masks='non-intersecting retained masks inform matching/containment; never selected; never bridge groups',whole_part_priority=True,crop_contact_policy='warning only; other conflicts may defer',noise_filter=dict(max_tiny_area=25,max_line_area=100,max_line_width=2,min_line_aspect=5,width_definition='min rotated rectangle of pixel centers plus one pixel'),consistent_triple_with_extras='select consistent original/local/SAM main; remaining non-noise fragments defer separately',independent_per_frame_and_crop=True,no_semantic_module=True),source_sha256=protected,crops={})
    stats=collections.Counter();reasons=collections.Counter();unique_selected=set();rois_selected=set();checks=0;group_sizes=collections.Counter();shape_cache={};unique_noise=set()
    for ji,job in enumerate(f['jobs']):
        key=job['key'];rel=ir['crops'][key];nodes=[];source_nodes=[]
        for mi,sid in enumerate(job['sources']):
            src=f['sources'][sid];arr,areas=pixels(sid);members=set(job['members'][mi]);nn=[]
            for lid,cid in src['mapping'].items():
                c=f['candidates'][cid];lid=int(lid)
                if cid not in members or c['decision']=='reject':continue
                row=rel['models'][mi]['labels'][str(lid)]
                n=dict(key=f'{mi}:{lid}',model=mi,label=lid,candidate_id=c['id'],source_index=sid,source_path=src['path'],source_box=src['box'],area=int(areas[lid]),eligible=row['decision']=='intersects',touches_crop=c['touches_crop'],touches_frame=c['touches_frame'],support=row)
                if c['id'] not in shape_cache:
                    if n['area']<=100:
                        yy,xx=np.nonzero(arr==lid)
                        rect=cv2.minAreaRect(np.column_stack((xx,yy)).astype(np.float32));ww,ll=sorted((float(rect[1][0])+1,float(rect[1][1])+1))
                        geometry=dict(width=ww,length=ll,aspect=ll/ww,bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
                    else:geometry=dict(width=None,length=None,aspect=None,bbox=None)
                    shape_cache[c['id']]=geometry
                n['geometry']=shape_cache[c['id']]
                n['noise_reason']=noise_reason(n['area'],n['geometry']['width'],n['geometry']['length'])
                nodes.append(n)
                if not n['noise_reason']:nn.append(n)
            source_nodes.append(nn)
        edges=[]
        for mi,mj in itertools.combinations(range(3),2):
            aa,ar=pixels(job['sources'][mi]);bb,br=pixels(job['sources'][mj]);ncol=len(br)
            hist=np.bincount((aa.ravel().astype(np.int64)*ncol+bb.ravel()),minlength=len(ar)*ncol).reshape(len(ar),ncol)
            ids_a=[n['label'] for n in source_nodes[mi]];ids_b=[n['label'] for n in source_nodes[mj]]
            if not ids_a or not ids_b:continue
            inter=hist[np.ix_(ids_a,ids_b)];union=ar[ids_a,None]+br[None,ids_b]-inter;iou=np.divide(inter,union,out=np.zeros(inter.shape,float),where=union>0)
            for ai,an in enumerate(source_nodes[mi]):
                for bi,bn in enumerate(source_nodes[mj]):
                    inter_n=int(inter[ai,bi])
                    if not inter_n:continue
                    area_a=an['area'];area_b=bn['area'];un=area_a+area_b-inter_n
                    mutual=bool(iou[ai,bi]==iou[ai].max() and iou[ai,bi]==iou[:,bi].max() and np.sum(iou[ai]==iou[ai,bi])==1 and np.sum(iou[:,bi]==iou[ai,bi])==1)
                    high=10*inter_n>=9*un;contain=20*inter_n>=19*min(area_a,area_b) and not high
                    link=2*inter_n>=min(area_a,area_b)
                    if link:
                        e=dict(a=an['key'],b=bn['key'],intersection=inter_n,union=un,iou=inter_n/un,coverage_a=inter_n/area_a,coverage_b=inter_n/area_b,mutual_best=mutual,near=high and mutual,containment=contain,group_link=True)
                        edges.append(e)
                        # Recount randomly sampled and threshold-adjacent relationships directly.
                        if (ji+ai+bi)%37==0 or any(abs(inter_n/v-t)<.002 for v,t in [(un,.9),(min(area_a,area_b),.95),(min(area_a,area_b),.5)]):
                            assert int(np.count_nonzero((aa==an['label'])&(bb==bn['label'])))==inter_n;checks+=1
        groups=choose(nodes,edges);node_map={n['key']:n for n in nodes};models=[]
        for mi in range(3):
            model_nodes=[n for n in nodes if n['model']==mi]
            models.append(dict(labels={str(n['label']):dict(status='noise' if n['noise_reason'] else 'no_intersection',group=None,reason=n['noise_reason'],warnings=[]) for n in model_nodes},selected_ids=[],deferred_ids=[],alternative_ids=[],noise_ids=[n['label'] for n in model_nodes if n['noise_reason']]))
            for n in model_nodes:
                if n['noise_reason']:
                    stats['noise_occurrences']+=1;stats['noise_intersects' if n['eligible'] else 'noise_no_intersection']+=1;unique_noise.add(n['candidate_id'])
        for g in groups:
            reasons[g['reason']]+=1;group_sizes[len(g['members'])]+=1;stats[g['status']+'_groups']+=1
            for nk in g['members']:
                n=node_map[nk];status='deferred' if g['selected'] is None else ('selected' if g['selected']==nk else 'alternative');mi=n['model'];lid=n['label']
                models[mi]['labels'][str(lid)]=dict(status=status,group=g['id'],reason=g['reason'],warnings=g['warnings'])
                models[mi][status+'_ids'].append(lid);stats[status+'_occurrences']+=1
                if status=='selected':unique_selected.add((job['uid'],n['candidate_id']));rois_selected.add(job['uid'])
        assert sum(len(m[t+'_ids']) for m in models for t in ['selected','deferred','alternative'])+sum(n['eligible'] and bool(n['noise_reason']) for n in nodes)==sum(len(m['eligible_ids']) for m in rel['models'])
        out['crops'][key]=dict(uid=job['uid'],ROI=job['ROI'],frame=job['frame'],nodes=nodes,groups=groups,models=models)
        if (ji+1)%75==0:print('JOBS',ji+1,'/',len(f['jobs']),dict(stats),flush=True)
    assert len(out['crops'])==len(f['jobs'])
    assert stats['selected_occurrences']+stats['deferred_occurrences']+stats['alternative_occurrences']+stats['noise_intersects']==ir['stats']['intersects']
    assert protected=={p:sha(Path(p)) for p in protected}
    out['stats']=dict(stats,unique_ROI_selected_candidate_pairs=len(unique_selected),ROI_with_selected=len(rois_selected),ROI_count=len(f['cases']),crops=len(out['crops']),reason_counts=dict(reasons),group_sizes=dict(group_sizes),unique_noise_candidates=len(unique_noise))
    save(R/'selection.json',out)
    assert 'CROP_TRUNCATION_UNCERTAIN' not in reasons
    save(R/'revision_changes.json',dict(scope='room0 first run of accepted r2 rules; no historical room2 example audit',current_stats=out['stats']))
    audit=dict(status='PASS',stats=out['stats'],rule=out['rule'],independent_pixel_recounts=checks,verified_source_masks=len(verified),input_files_unchanged=True,no_new_inference=True,no_map_write=True,no_GT_used=True,seconds=round(time.time()-start,2));save(R/'audit.json',audit)
    with zipfile.ZipFile(R/'selection_data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['selection.json','audit.json','revision_changes.json']:z.write(R/name,name)
    save(R/'receipt.json',dict(sha256=sha(R/'selection_data.zip'),bytes=(R/'selection_data.zip').stat().st_size));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
