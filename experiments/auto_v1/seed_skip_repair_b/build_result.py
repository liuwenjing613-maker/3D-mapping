"""User-confirmed single-frame B decisions and auditable source-view images."""
from pathlib import Path
import copy,json,shutil,zipfile,time
import numpy as np
from PIL import Image,ImageDraw
from measure_inputs import sha


def main():
    started=time.perf_counter();code=Path(__file__).resolve().parent
    cfg=json.loads((code/'config.json').read_text());out=Path(cfg['output_directory']);inp=Path(cfg['input_directory'])
    assert cfg['minimum_p']==cfg['minimum_r']==.95 and not cfg['conflict_gate'] and cfg['minimum_support_gate'] is None
    verified=json.loads((out/'measurement_verification.json').read_text())
    assert verified['status']=='PASS' and sha(out/'measurements.json')==verified['measurements_sha256']
    for path,digest in verified['source_sha256'].items():assert sha(path)==digest,('input changed',path)
    data=json.loads((out/'measurements.json').read_text());manifest=json.loads((inp/'seed_manifest.json').read_text())
    groups={g['id']:g for g in manifest['groups']}
    records=[]
    for m in data['primaries']:
        inter=m['intersection'];s=m['support_count'];i=m['instance_visible_count']
        skip=bool(s>0 and i>0 and 20*inter>=19*s and 20*inter>=19*i)
        assert skip==m['both95']
        records.append({**m,'decision':'skip_repair' if skip else 'retain','reason':'both_coverage_at_least_95' if skip else
                        ('no_assigned_instance' if m['instance_id'] is None else 'coverage_below_95'),
                        'area':groups[m['group_id']]['primary']['area'],
                        'conflict_fraction':m['native_state_counts']['conflict']/s})
    skipped=[r for r in records if r['decision']=='skip_repair'];retained=[r for r in records if r['decision']=='retain']
    assert len(records)==56 and len(skipped)==11 and len(retained)==45
    def save(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stats=dict(input=56,skipped=len(skipped),retained=len(retained),source_frames=len({r['frame'] for r in records}),
               skipped_with_native_conflicts=sum(r['native_state_counts']['conflict']>0 for r in skipped),
               smallest_input_support=min(r['support_count'] for r in records),
               smallest_skipped_support=min(r['support_count'] for r in skipped))
    # Decisions are frozen before rendering; no human selection or inferred visual judgment enters the rule.
    result=dict(schema='auto-v1-skip-repair-B-single-frame-result-v1',config=cfg,stats=stats,records=records,
                measurement_sha256=verified['measurements_sha256'],config_sha256=sha(code/'config.json'),
                decision_source='user-confirmed single-source-frame p/r rule')
    save('classification.json',result)
    save('selection_binary.json',dict(schema='auto-v1-skip-repair-B-selection-v1',counts=dict(skipped=11,retained=45),
        input_selection_sha256=sha(inp/'selection_binary.json'),classification_sha256=sha(out/'classification.json'),
        skipped=skipped,retained=retained,tracking_requested=False,map_write=False))
    manifest_out=copy.deepcopy(manifest)
    manifest_out.update(groups=[groups[r['group_id']] for r in retained],skip_repair_B_config=cfg,
                        skip_repair_B_group_ids=[r['group_id'] for r in skipped],
                        skip_repair_B_source_manifest_sha256=sha(inp/'seed_manifest.json'),
                        status='single_frame_B_screening_only',tracking_requested=False,map_write=False)
    save('seed_manifest.json',manifest_out)
    if not (out/'masks').exists():(out/'masks').symlink_to((inp/'masks').resolve(),target_is_directory=True)
    for r in retained:assert sha(out/groups[r['group_id']]['primary']['mask'])==r['mask_sha256']
    # Same source view for all three images. The instance extent is never clipped to the seed.
    with np.load(cfg['surface']) as z:labels=z['instance_id']
    with np.load(cfg['evidence']) as z:state=z['state']
    camera=json.loads(Path(cfg['camera_config']).read_text());scene=Path(camera['source']['scene_root'])
    original=Path(cfg['original_directory'])
    with np.load(original/'surface_supports.npz') as z:ids=z['indices'];offsets=z['offsets']
    preview=out/'previews';preview.mkdir(exist_ok=True);visual_rows=[];rgb_cache={};visibility={};render_sources={}
    def render(base,name,box):
        image=Image.fromarray(base).crop(box);image.thumbnail((900,600));image.save(out/name,quality=92)
    for rec in records:
        s=groups[rec['group_id']]['primary'];gid=rec['group_id'];frame=rec['frame'];idx=s['index']
        if frame not in rgb_cache:
            rp=scene/camera['source']['rgb_pattern'].format(frame=frame)
            rgb_cache[frame]=np.asarray(Image.open(rp).convert('RGB'));render_sources[str(rp)]=sha(rp)
            vp=Path(cfg['visibility_directory'])/f'frame{frame:06d}_visible.npz'
            with np.load(vp) as z:visibility[frame]=(z['vertex_index'],z['uv'])
        rgb=rgb_cache[frame];visible,uv=visibility[frame]
        support=ids[offsets[idx]:offsets[idx+1]]
        instance=visible[labels[visible]==rec['instance_id']] if rec['instance_id'] is not None else np.array([],dtype=np.int32)
        common=np.intersect1d(support,instance,assume_unique=True)
        seed_only=np.setdiff1d(support,instance,assume_unique=True)
        instance_only=np.setdiff1d(instance,support,assume_unique=True)
        assert (len(support),len(instance),len(common))==(rec['support_count'],rec['instance_visible_count'],rec['intersection'])
        mask=np.asarray(Image.open(original/s['mask']))>0
        mx0,my0,mx1,my1=s['bbox'];box=[mx0,my0,mx1,my1]
        if len(instance):
            pts=uv[np.searchsorted(visible,instance)]
            box=[min(mx0,int(pts[:,0].min())),min(my0,int(pts[:,1].min())),max(mx1,int(pts[:,0].max())+1),max(my1,int(pts[:,1].max())+1)]
        x0,y0,x1,y1=box;box=(max(0,x0-35),max(0,y0-35),min(1200,x1+35),min(680,y1+35))
        images={}
        seed_img=(rgb*.75).astype(np.uint8);seed_img[mask]=(rgb[mask]*.55+np.array([40,210,160])*.45).astype(np.uint8)
        name=f'previews/{gid}_seed.jpg';render(seed_img,name,box);images['seed']=name
        # Use unique projected pixels for display; all computations above remain unique 3D vertex counts.
        def overlay(points_by_color):
            arr=(rgb*.60).astype(np.uint8)
            for points,color in points_by_color:
                if not len(points):continue
                positions=np.searchsorted(visible,points);assert np.array_equal(visible[positions],points)
                xy=uv[positions];pix=np.unique(xy[:,1]*1200+xy[:,0]);y=pix//1200;x=pix%1200
                for dy,dx in [(0,0),(0,1),(1,0),(0,-1),(-1,0)]:
                    xx=np.clip(x+dx,0,1199);yy=np.clip(y+dy,0,679)
                    arr[yy,xx]=(arr[yy,xx]*.25+np.array(color)*.75).astype(np.uint8)
            return arr
        name=f'previews/{gid}_instance.jpg';render(overlay([(instance,(90,165,255))]),name,box);images['instance']=name
        name=f'previews/{gid}_difference.jpg';render(overlay([(instance_only,(90,165,255)),(seed_only,(255,165,55)),(common,(35,225,152))]),name,box);images['difference']=name
        conflicts=support[state[support]==3]
        name=f'previews/{gid}_conflict.jpg';render(overlay([(support,(80,110,115)),(conflicts,(255,105,80))]),name,box);images['conflict']=name
        name=f'previews/{gid}_full.jpg';Image.fromarray(seed_img).save(out/name,quality=90);images['full']=name
        tight=(max(0,mx0-20),max(0,my0-20),min(1200,mx1+20),min(680,my1+20))
        name=f'previews/{gid}_thumb.jpg';render(seed_img,name,tight);images['thumb']=name
        visual_rows.append({**rec,'images':images,'display_box':list(box)})
    save('review_data.json',{**result,'records':visual_rows,'classification_sha256':sha(out/'classification.json'),
                            'computation_seconds':data['elapsed_seconds'],
                            'render_note':'图片中投影点扩大1像素以便观察，未用于数值计算；三张对照图共享视角与裁剪范围，包含完整可见实例。'})
    save('render_sources.json',render_sources)
    for name in ['index.html','review.js','review.css']:shutil.copyfile(code/name,out/name)
    source_sel=json.loads((inp/'selection_binary.json').read_text())
    assert {r['group_id'] for r in records}=={r['group_id'] for r in source_sel['retained']}
    assert all(r['original_track_id']==int(r['group_id'][1:]) for r in records)
    save('verification.json',dict(status='PASS',input_verification_sha256=sha(out/'measurement_verification.json'),
        config_sha256=sha(code/'config.json'),code_sha256={p.name:sha(p) for p in code.iterdir() if p.suffix in ['.py','.json']},
        checks=['Only 56 currently retained seeds classified; no excluded seed reintroduced',
                'User-confirmed single-frame inclusive p/r >= 0.95 rule, with no extra conflict or sample gate',
                '11 skipped and 45 retained form a disjoint complete partition',
                'All mask hashes and original IDs preserved; no mask pixels changed',
                'All rendered support, instance and intersection counts independently recounted',
                'Instance denominator and display extent cover the whole source-visible instance',
                'No second-frame decisions, candidate replacement, E1 run or GT filtering'],
        elapsed_seconds=time.perf_counter()-started,
        output_sha256={n:sha(out/n) for n in ['classification.json','selection_binary.json','seed_manifest.json','review_data.json','render_sources.json']}))
    package=['classification.json','selection_binary.json','seed_manifest.json','review_data.json','render_sources.json','verification.json','measurement_verification.json','measurements.json','index.html','review.js','review.css']
    with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in package:z.write(out/name,name)
        for p in sorted(preview.glob('*.jpg')):z.write(p,p.relative_to(out).as_posix())
    receipt=dict(sha256=sha(out/'review_bundle.zip'),bytes=(out/'review_bundle.zip').stat().st_size,stats=stats)
    save('delivery_receipt.json',receipt);print(json.dumps(receipt));print('SKIP:',','.join(r['group_id'] for r in skipped))


if __name__=='__main__':main()
