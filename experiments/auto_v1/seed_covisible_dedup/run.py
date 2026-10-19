"""Audited experiment on the 72 unchanged, semantically retained final seeds."""
from pathlib import Path
import hashlib, json, time, sys, copy
from collections import Counter
import numpy as np
from PIL import Image
from rules import pair_metrics, order_key, fixed_primary_groups


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.perf_counter()
    code=Path(__file__).resolve().parent
    cfg=json.loads((code/'config.json').read_text())
    assert (cfg['minimum_visible_fraction_each'],cfg['minimum_visible_vertices_each'],cfg['minimum_common_visible_iou'])==(.3,20,.8)
    inp=Path(cfg['input_directory']);orig=Path(cfg['original_directory']);out=Path(cfg['output_directory'])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Completed experiment already exists; use a new output directory.')
    locked={}
    def lock(p,expected=None):
        value=sha(p)
        if expected is not None:assert value==expected,('hash changed',str(p))
        locked[str(p)]=value
        return value
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lock(inp/'selection_binary.json',cfg['input_selection_sha256'])
    lock(inp/'seed_manifest.json',cfg['input_manifest_sha256'])
    lock(orig/'grouping.json',cfg['original_grouping_sha256'])
    sel=json.loads((inp/'selection_binary.json').read_text())
    manifest=json.loads((inp/'seed_manifest.json').read_text())
    old=json.loads((orig/'grouping.json').read_text())
    audit=json.loads((orig/'audit.json').read_text())
    assert audit['status']=='PASS'
    for name in ['surface_supports.npz','pair_metrics.npz','seed_manifest.json']:
        lock(orig/name,audit['output_sha256'][name])
    rows=sel['retained'];bygroup={g['id']:g for g in manifest['groups']}
    assert sel['counts']==dict(excluded=18,retained=72)
    assert len(rows)==len(bygroup)==72 and set(bygroup)=={r['group_id'] for r in rows}
    assert not set(bygroup)&{r['group_id'] for r in sel['excluded']}
    with np.load(orig/'surface_supports.npz') as z:indices=z['indices'];offsets=z['offsets']
    with np.load(orig/'pair_metrics.npz') as z:old_inter=z['intersection']
    supports=[];seeds=[];masks=[]
    for row in rows:
        g=bygroup[row['group_id']];s=g['primary'];idx=s['index']
        assert g['original_track_id']==row['original_track_id']==int(g['id'][1:])
        assert s['id']==row['mask_id'] and s==old['seeds'][idx]
        assert s['frame']==row['frame'] and s['mask_sha256']==row['mask_sha256']
        lock(inp/s['mask'],s['mask_sha256'])
        a=np.asarray(Image.open(inp/s['mask']))>0
        assert a.shape==(680,1200) and int(a.sum())==s['area']
        assert hashlib.sha256(s['frame'].to_bytes(4,'little')+np.packbits(a.reshape(-1)).tobytes()).hexdigest()==s['digest']
        support=indices[offsets[idx]:offsets[idx+1]]
        assert len(support)==s['support_count'] and np.all(support[:-1]<support[1:])
        assert hashlib.sha256(support.tobytes()).hexdigest()==s['support_sha256']
        seeds.append(s);supports.append(support);masks.append(a)
    sources=old['source_sha256']
    geom=Path(cfg['surface']);camera_path=Path(cfg['camera_config']);module=Path(cfg['projection_module'])
    for p in [geom,camera_path,module]:lock(p,sources[str(p)])
    camcfg=json.loads(camera_path.read_text());camera=camcfg['camera'];scene=Path(camcfg['source']['scene_root'])
    traj=scene/camcfg['source']['trajectory'];lock(traj,sources[str(traj)])
    poses=np.loadtxt(traj).reshape(-1,4,4)
    with np.load(geom) as z:xyz=z['xyz_m']
    assert len(xyz)==old['visibility_rule']['vertex_count']
    sys.path.insert(0,str(module.parent));import common
    assert common.DEPTH_TOLERANCE==old['visibility_rule']['depth_tolerance']==.02
    visible={};frames=sorted({s['frame'] for s in seeds})
    t=time.perf_counter()
    for number,frame in enumerate(frames):
        p=Path(cfg['visibility_directory'])/f'frame{frame:06d}_visible.npz';lock(p,sources[str(p)])
        depthp=scene/camcfg['source']['depth_pattern'].format(frame=frame);lock(depthp,sources[str(depthp)])
        depth=np.asarray(Image.open(depthp),dtype=np.float64)/camera['depth_png_units_per_meter']
        valid,uv=common.project_visible(xyz,poses[frame],depth,camera)
        with np.load(p) as z:cachedidx=z['vertex_index'];cacheduv=z['uv']
        idx=np.flatnonzero(valid).astype(np.int32)
        assert np.array_equal(idx,cachedidx) and np.array_equal(uv[idx],cacheduv),('cache mismatch',frame)
        for i,s in enumerate(seeds):
            if s['frame']==frame:
                projected=idx[masks[i][cacheduv[:,1],cacheduv[:,0]]]
                assert np.array_equal(projected,supports[i]),('support mismatch',s['id'])
        visible[frame]=valid
        if number%10==0:print('Verified source view',number+1,'/',len(frames),flush=True)
    geometry_seconds=time.perf_counter()-t
    t=time.perf_counter()
    restricted={(i,frame):a[visible[frame][a]] for i,a in enumerate(supports) for frame in frames}
    pairs=[];lookup={}
    for i,a in enumerate(supports):
        for j in range(i+1,len(supports)):
            b=supports[j]
            av=restricted[i,seeds[j]['frame']];bv=restricted[j,seeds[i]['frame']]
            m=pair_metrics(a,b,av,bv)
            # Each support is already source-visible, so restricting visibility cannot remove A intersect B.
            assert m['intersection']==m['original_intersection']==int(old_inter[seeds[i]['index'],seeds[j]['index']])
            assert m['original_iou']<=.25, 'Input final primaries should already pass the old deduplication.'
            m.update(a=rows[i]['group_id'],b=rows[j]['group_id'],frame_a=seeds[i]['frame'],frame_b=seeds[j]['frame'])
            pairs.append(m);lookup[i,j]=m
    ordered=sorted(range(len(seeds)),key=lambda i:order_key(seeds[i]))
    grouped=fixed_primary_groups(ordered,lambda i,j:lookup[min(i,j),max(i,j)]['duplicate'])
    group_of={i:members[0] for members in grouped for i in members}
    grouping=[]
    for members in grouped:
        p=members[0]
        grouping.append(dict(primary=rows[p]['group_id'],members=[rows[i]['group_id'] for i in members],
                             support_count=seeds[p]['support_count']))
        for j in members[1:]:assert lookup[min(p,j),max(p,j)]['duplicate'] and seeds[p]['support_count']>=seeds[j]['support_count']
    records=[]
    for i,row in enumerate(rows):
        p=group_of[i]
        decision='retain' if p==i else 'exclude'
        records.append({**row,'semantic_decision':row['decision'],'decision':decision,
                        'primary_group_id':rows[p]['group_id'],'dedup_decision':decision,
                        'support_count':seeds[i]['support_count'],'area':seeds[i]['area'],
                        'source_model':seeds[i]['representative']['model']})
    retained=[r for r in records if r['dedup_decision']=='retain'];excluded=[r for r in records if r['dedup_decision']=='exclude']
    decision_seconds=time.perf_counter()-t
    stats=dict(input=72,retained=len(retained),excluded=len(excluded),pairs=len(pairs),
               duplicate_pairs=sum(p['duplicate'] for p in pairs),sufficient_evidence_pairs=sum(p['evidence_sufficient'] for p in pairs),
               reason_counts=dict(Counter(p['reason'] for p in pairs)),source_frames=len(frames))
    assert len(pairs)==2556 and len(retained)+len(excluded)==72
    assert all(r['decision']=='retain' for r in retained) and all(r['decision']=='exclude' for r in excluded)
    save('pair_results.json',dict(config=cfg,stats=stats,pairs=pairs,groups=grouping,records=records))
    save('selection_binary.json',dict(schema='auto-v1-covisible-binary-selection-v1',input_selection_sha256=cfg['input_selection_sha256'],
          counts=dict(retained=len(retained),excluded=len(excluded)),retained=retained,excluded=excluded,groups=grouping))
    result_manifest=copy.deepcopy(manifest)
    result_manifest.update(groups=[bygroup[r['group_id']] for r in retained],dedup_config=cfg,dedup_groups=grouping,
                           dedup_input_manifest_sha256=cfg['input_manifest_sha256'],dedup_excluded_group_ids=[r['group_id'] for r in excluded],
                           status='dedup_experiment',tracking_requested=False,map_write=False)
    save('seed_manifest.json',result_manifest)
    if not (out/'masks').exists():(out/'masks').symlink_to((inp/'masks').resolve(),target_is_directory=True)
    for r in retained:assert sha(out/bygroup[r['group_id']]['primary']['mask'])==r['mask_sha256']
    for p,h in locked.items():assert sha(p)==h,('input changed during experiment',p)
    save('input_freeze.json',dict(input_hashes=locked,code_hashes={p.name:sha(p) for p in sorted(code.iterdir()) if p.suffix in ['.py','.json']}))
    save('verification.json',dict(status='PASS',checks=['72 input mask bytes, full binary digests and support hashes match frozen inputs',
         '47 source visibility caches independently reprojected from unchanged depth, pose and TSDF geometry',
         'Every source-view support independently reconstructed from its complete mask',
         'All 2556 original intersections equal archived matrix and mutually visible intersections',
         'No transitive grouping; every excluded seed directly passes against its kept representative',
         'Original IDs and mask files preserved; excluded semantic seeds not reintroduced',
         'Only coordinates and visibility read; no existing instance IDs, GT, human decisions or tracking used'],
         geometry_validation_seconds=geometry_seconds,decision_seconds=decision_seconds,total_seconds=time.perf_counter()-started,
         output_sha256={n:sha(out/n) for n in ['pair_results.json','selection_binary.json','seed_manifest.json','input_freeze.json']}))
    save('complete.json',dict(status='PASS',stats=stats,verification_sha256=sha(out/'verification.json')))
    print(json.dumps(stats),flush=True)
    print('DUPLICATE PAIRS',json.dumps([p for p in pairs if p['duplicate']]),flush=True)


if __name__=='__main__':main()
