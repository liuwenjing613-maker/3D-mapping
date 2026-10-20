"""Semantic SAM routing precedes ordinary overmerge/intersection/noise/version gates."""
from pathlib import Path
import collections,hashlib,itertools,json,time,zipfile,copy
import numpy as np
from PIL import Image
from selection_rules import choose

ROOT=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office1/semantic_route')
FILTER=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office1/overmerge')
INTER=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office1/intersection/web/intersection.json')
PREVIOUS=Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/office1/versions/selection.json')
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def edge(a,b,inter,mutual=False):
    union=a['area']+b['area']-inter
    return dict(a=a['key'],b=b['key'],intersection=int(inter),union=int(union),iou=float(inter/union),coverage_a=float(inter/a['area']),coverage_b=float(inter/b['area']),mutual_best=bool(mutual),near=bool(10*inter>=9*union and mutual),containment=bool(20*inter>=19*min(a['area'],b['area']) and 10*inter<9*union),group_link=bool(2*inter>=min(a['area'],b['area'])))

def main():
    started=time.time();f=read(FILTER/'web/data.json');ir=read(INTER);previous=read(PREVIOUS);semantic=read(ROOT/'semantic.json');hashes=read(FILTER/'input_hashes.json')
    protected={str(p):sha(p) for p in [FILTER/'web/data.json',INTER,PREVIOUS,ROOT/'semantic.json']}
    cache=collections.OrderedDict();verified=set();checks=0
    def pixels(sid):
        if sid in cache:cache.move_to_end(sid);return cache[sid]
        src=f['sources'][sid];p=Path(src['path'])
        if str(p) not in verified:assert sha(p)==hashes[str(p)];verified.add(str(p))
        a=np.asarray(Image.open(p),np.int32);x0,y0,x1,y1=src['box'];assert a.shape==(y1-y0,x1-x0)
        counts=np.bincount(a.ravel());full=np.zeros((680,1200),np.int32);full[y0:y1,x0:x1]=a;cache[sid]=(full,counts)
        if len(cache)>18:cache.popitem(last=False)
        return full,counts
    rule=copy.deepcopy(previous['rule']);rule['no_semantic_module']=False;rule['semantic_route']=semantic['config']
    out=dict(schema='mask-version-selection-v3-semantic-sam',rule=rule,source_sha256=protected,semantic_model=semantic['model'],semantic_terms=semantic['terms'],crops={})
    stats=collections.Counter();reasons=collections.Counter();families=collections.Counter();bypasses=collections.Counter();old_selected_changes=collections.Counter();unique_selected=set();unique_semantic=set();roi_selected=set();unchanged_crops=0
    for job_index,job in enumerate(f['jobs']):
        key=job['key'];rel=ir['crops'][key];old=previous['crops'][key];old_nodes={n['key']:n for n in old['nodes']};nodes=[];by_source=[];by_key={}
        for mi,sid in enumerate(job['sources']):
            src=f['sources'][sid];_,areas=pixels(sid);members=set(job['members'][mi]);current=[]
            for label,cid in src['mapping'].items():
                if cid not in members:continue
                lid=int(label);c=f['candidates'][cid];nk=f'{mi}:{lid}';support=rel['models'][mi]['labels'][label]
                n=copy.deepcopy(old_nodes[nk]) if nk in old_nodes else dict(key=nk,model=mi,label=lid,candidate_id=c['id'],source_index=sid,source_path=src['path'],source_box=src['box'],area=int(areas[lid]),eligible=support['expanded']>0,touches_crop=c['touches_crop'],touches_frame=c['touches_frame'],support=support,geometry=dict(width=None,length=None,aspect=None,bbox=None),noise_reason=None)
                n['upstream_overmerge']=c['decision']=='reject';n['semantic']=semantic['candidates'][c['id']] if mi==2 else None
                n['semantic_hit']=bool(mi==2 and n['semantic']['hit']);nodes.append(n);current.append(n);by_key[nk]=n
            by_source.append(current)
        hits={n['key'] for n in nodes if n['semantic_hit']};replaced=set();replacement_edges=[];pair_hist={}
        # Correspondence is computed on the unfiltered partitions. It only withdraws
        # CropFormer counterparts; it is never an acceptance gate for semantic SAM.
        for mi,mj in itertools.combinations(range(3),2):
            aa,ar=pixels(job['sources'][mi]);bb,br=pixels(job['sources'][mj]);cols=len(br)
            hist=np.bincount(aa.ravel().astype(np.int64)*cols+bb.ravel(),minlength=len(ar)*cols).reshape(len(ar),cols);pair_hist[(mi,mj)]=hist
            if mj==2 and hits:
                for a in by_source[mi]:
                    for b in by_source[mj]:
                        if b['key'] not in hits:continue
                        inter=int(hist[a['label'],b['label']])
                        if inter and 2*inter>=min(a['area'],b['area']):
                            replaced.add(a['key']);e=edge(a,b,inter);e['semantic_replacement']=True;replacement_edges.append(e)
                            assert int(np.count_nonzero((aa==a['label'])&(bb==b['label'])))==inter;checks+=1
        ordinary={n['key'] for n in nodes if n['key'] not in hits|replaced and not n['upstream_overmerge'] and not n['noise_reason']}
        generic_nodes=[n for n in nodes if n['key'] in ordinary];edges=[]
        for mi,mj in itertools.combinations(range(3),2):
            aa=[n for n in by_source[mi] if n['key'] in ordinary];bb=[n for n in by_source[mj] if n['key'] in ordinary]
            if not aa or not bb:continue
            hist=pair_hist[(mi,mj)];inter=hist[np.ix_([n['label'] for n in aa],[n['label'] for n in bb])]
            union=np.array([n['area'] for n in aa])[:,None]+np.array([n['area'] for n in bb])[None,:]-inter;iou=inter/union
            for ai,a in enumerate(aa):
                for bi,b in enumerate(bb):
                    num=int(inter[ai,bi])
                    if not num or 2*num<min(a['area'],b['area']):continue
                    mutual=iou[ai,bi]==iou[ai].max() and iou[ai,bi]==iou[:,bi].max() and np.sum(iou[ai]==iou[ai,bi])==1 and np.sum(iou[:,bi]==iou[ai,bi])==1
                    edges.append(edge(a,b,num,mutual))
        groups=choose(generic_nodes,edges)
        direct_groups=[]
        for i,nk in enumerate(sorted(hits),1):
            n=by_key[nk];refs=[e for e in replacement_edges if e['b']==nk]
            direct_groups.append(dict(id=f'S{i:03d}',members=[nk],selected=nk,status='selected',reason='SEMANTIC_SAM_DIRECT',warnings=[],relations=[],references=refs,parent_group=f'S{i:03d}',original_members=[nk],semantic=n['semantic']))
        groups=direct_groups+groups;models=[]
        for mi in range(3):
            models.append(dict(labels={},selected_ids=[],deferred_ids=[],alternative_ids=[],noise_ids=[],overmerge_ids=[],no_intersection_ids=[],semantic_ids=[],semantic_replaced_ids=[]))
        for n in nodes:
            if n['key'] in hits:status,reason='selected','SEMANTIC_SAM_DIRECT'
            elif n['key'] in replaced:status,reason='alternative','SEMANTIC_CROPFORMER_REPLACED'
            elif n['upstream_overmerge']:status,reason='overmerge','OVERMERGE'
            elif n['noise_reason']:status,reason='noise',n['noise_reason']
            else:status,reason='no_intersection',None
            models[n['model']]['labels'][str(n['label'])]=dict(status=status,group=None,reason=reason,warnings=[])
        for g in groups:
            for nk in g['members']:
                n=by_key[nk];status='deferred' if not g['selected'] else ('selected' if nk==g['selected'] else 'alternative')
                models[n['model']]['labels'][str(n['label'])]=dict(status=status,group=g['id'],reason=g['reason'],warnings=g['warnings'])
            reasons[g['reason']]+=1;stats[g['status']+'_groups']+=1
        for nk in replaced:
            n=by_key[nk];row=models[n['model']]['labels'][str(n['label'])];row['semantic_matches']=[e['b'] for e in replacement_edges if e['a']==nk]
        for n in nodes:
            m=models[n['model']];row=m['labels'][str(n['label'])];m[row['status']+'_ids'].append(n['label']);stats[row['status']+'_occurrences']+=1
            if row['status']=='selected':unique_selected.add((job['uid'],n['candidate_id']));roi_selected.add(job['uid'])
            if n['key'] in hits:
                assert row['status']=='selected';m['semantic_ids'].append(n['label']);families[n['semantic']['target_family']]+=1;unique_semantic.add(n['candidate_id'])
                bypasses['overmerge']+=int(n['upstream_overmerge']);bypasses['no_intersection']+=int(n['support']['expanded']==0);bypasses['noise']+=int(bool(n['noise_reason']))
                before=old['models'][2]['labels'].get(str(n['label']),{});bypasses['previous_deferred']+=int(before.get('status')=='deferred')
            if n['key'] in replaced:m['semantic_replaced_ids'].append(n['label']);assert row['status']=='alternative'
        for mi,m in enumerate(old['models']):
            for lid in m['selected_ids']:old_selected_changes[models[mi]['labels'][str(lid)]['status']]+=1
        if not hits:
            for mi,m in enumerate(old['models']):
                for lid,row in m['labels'].items():assert models[mi]['labels'][lid]['status']==row['status'],(key,mi,lid)
            unchanged_crops+=1
        assert sum(len(m['labels']) for m in models)==sum(len(x) for x in job['members'])
        assert all(not (set(g['members'])&replaced) for g in groups)
        out['crops'][key]=dict(uid=job['uid'],ROI=job['ROI'],frame=job['frame'],nodes=nodes,groups=groups,models=models,semantic_replacement_edges=replacement_edges)
        if (job_index+1)%100==0:print('ROUTE',job_index+1,'/',len(f['jobs']),dict(stats),flush=True)
    assert protected=={p:sha(Path(p)) for p in protected}
    stats['semantic_selected_occurrences']=sum(families.values());stats['generic_selected_occurrences']=stats['selected_occurrences']-sum(families.values());stats['semantic_replaced_CF_occurrences']=sum(len(m['semantic_replaced_ids']) for c in out['crops'].values() for m in c['models'])
    out['stats']=dict(stats,semantic_family_occurrences=dict(families),semantic_unique_candidates=len(unique_semantic),unique_ROI_selected_candidate_pairs=len(unique_selected),ROI_with_selected=len(roi_selected),ROI_count=len(f['cases']),crops=len(out['crops']),reason_counts=dict(reasons),semantic_bypassed_prior_rules=dict(bypasses),previous_selected_statuses=dict(old_selected_changes),unchanged_crops_without_semantic_hit=unchanged_crops)
    save(ROOT/'selection.json',out)
    audit=dict(status='PASS',stats=out['stats'],all_SAM_classified_before_downstream_filters=True,semantic_hits_all_selected=True,cropformer_matches_removed_from_generic_processing=True,ordinary_rules_source_sha256=sha(Path(__file__).parent/'selection_rules.py'),ordinary_rules_reference_sha256=sha(Path('/home/chenkejun/CVPR/experiments/auto_v2_replica6_20261019/office1/versions/selection_rules.py')),independent_replacement_pixel_recounts=checks,verified_source_masks=len(verified),source_files_unchanged=True,no_GT_used=True,no_map_write=True,no_new_segmentation_inference=True,seconds=round(time.time()-started,2))
    assert audit['ordinary_rules_source_sha256']==audit['ordinary_rules_reference_sha256'];save(ROOT/'audit.json',audit)
    with zipfile.ZipFile(ROOT/'route_data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['selection.json','semantic.json','audit.json']:z.write(ROOT/name,name)
    save(ROOT/'receipt.json',dict(sha256=sha(ROOT/'route_data.zip'),bytes=(ROOT/'route_data.zip').stat().st_size));print('COMPLETE',json.dumps(audit),flush=True)

if __name__=='__main__':main()
