"""Render the actual prefix inputs, ROI prompts, and all five filtering decisions."""
from online_common import *
import colorsys
import shutil
import zipfile
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W=R/'review'
ASSETS=W/'assets'
ASSETS.mkdir(parents=True,exist_ok=True)
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
image_cache={}
overlay_cache={}

def save_image(name, array, width=780):
    path=ASSETS/name
    if not path.exists():
        im=Image.fromarray(array.astype(np.uint8))
        if im.width>width:im.thumbnail((width,10000))
        im.save(path,quality=84,method=4)
    return 'assets/'+name

def rgb_frame(fid):
    if fid not in image_cache:
        assert fid in FIDS
        image_cache[fid]=np.asarray(Image.open(R/f'inputs/scene/results/frame{fid:06d}.jpg').convert('RGB'))
    return image_cache[fid]

def overlay(rgb, labels, name, binary=False):
    if (ASSETS/name).exists():
        return 'assets/'+name
    colors=np.zeros((int(labels.max())+1,3),np.uint8)
    if binary:colors[1]=[30,205,160]
    else:
        for k in range(1,len(colors)):
            colors[k]=np.array(colorsys.hsv_to_rgb((k*.61803398875)%1,.70,1.))*255
    valid=labels>0
    image=rgb.copy()
    image[valid]=(image[valid]*.47+colors[labels[valid]]*.53).astype(np.uint8)
    edge=np.zeros(labels.shape,bool)
    edge[1:]|=labels[1:]!=labels[:-1];edge[:,1:]|=labels[:,1:]!=labels[:,:-1]
    image[edge&valid]=[255,255,255]
    im=Image.fromarray(image)
    if not binary:
        draw=ImageDraw.Draw(im)
        for k in np.unique(labels[valid]):
            yy,xx=np.nonzero(labels==k)
            if len(xx)<160:continue
            middle=len(xx)//2
            draw.text((int(xx[middle]),int(yy[middle])),str(int(k)),font=FONT,fill='white',stroke_width=2,stroke_fill='black')
    return save_image(name,np.asarray(im))

def model_assets(job):
    rgb=np.asarray(Image.open(job['input_file']).convert('RGB'))
    paths=[Path(job['baseline_file']),R/'frontend/cropformer'/(job['key']+'.png'),R/'frontend/sam2_1'/(job['key']+'.png')]
    result=[]
    for p in paths:
        key=job['RGB_sha256']+sha(p)
        if key not in overlay_cache:
            labels=np.asarray(Image.open(p)).astype(np.int32)
            assert labels.shape==rgb.shape[:2]
            overlay_cache[key]=overlay(rgb,labels,'model_'+hashlib.sha256(key.encode()).hexdigest()[:20]+'.webp')
        result.append(overlay_cache[key])
    return result

def main():
    started=time.time()
    assert read(R/'selection_complete.json')['status']=='PASS'
    mapping=read(R/'mapping_complete.json')
    catalog=read(R/'catalog/room0/candidate_manifest.json')
    frontend=read(R/'frontend/input_manifest.json')
    selected=read(R/'v2/selection.json')
    measurements=read(R/'v2/measurements.json')
    semantic=read(R/'v2/semantic/classification.json')
    seeds=read(R/'v1/seed_manifest.json')
    source_groups={g['id']:g for g in read(R/'v1/grouping.json')['groups']}
    kept=read(R/'v2/seed_manifest.json')
    jobs={j['key']:j for j in frontend['jobs']}
    rois=[]
    for case in frontend['cases']:
        views=[]
        for view in case['views']:
            fid=int(view['frame']);assert fid in FIDS
            rgb=rgb_frame(fid)
            alarm=np.asarray(Image.open(R/'frontend/source'/view['source_alarm']).convert('RGB'))
            shown=rgb.copy()
            for channel,color in [(0,[0,0,0]),(1,[255,225,0]),(2,[255,0,0])]:
                hit=alarm[:,:,channel]>0
                shown[hit]=color
            box=view['source_crop']
            canvas=Image.fromarray(shown);draw=ImageDraw.Draw(canvas)
            draw.rectangle(tuple(box),outline=(25,210,170),width=4)
            alarmkey=hashlib.sha256((view['source_alarm']+str(box)).encode()).hexdigest()[:16]
            alarmfile=save_image('roi_'+alarmkey+'.webp',np.asarray(canvas))
            variants=[]
            for key in view['variants']:
                job=jobs[key]
                variants.append(dict(key=key,strategy=job['strategy'],crop=job['crop_xyxy'],models=model_assets(job),
                    rgb=save_image('crop_'+job['RGB_sha256'][:18]+'.webp',np.asarray(Image.open(job['input_file']).convert('RGB')))))
            views.append(dict(frame=fid,processed=fid//5+1,group=view['trigger_group'],entry=view['source_entry'],
                rgb=save_image(f'frame_{fid:06d}.webp',rgb),alarm=alarmfile,variants=variants,recommended=view['recommended']))
        rois.append(dict(uid=case['uid'],name=case['ROI'],aliases=case['aliases'],counts=case['counts'],
                         rankings=case['rankings'],views=views))
    decision={r['group_id']:r for r in selected['records']}
    coverage={r['group_id']:r for r in measurements['coverage']}
    semantics={r['group_id']:r for r in semantic['records']}
    items=[]
    (W/'masks').mkdir(exist_ok=True)
    for g in seeds['groups']:
        gid=g['id'];p=g['primary'];fid=int(p['frame']);assert fid in FIDS
        maskpath=R/'v1'/p['mask'];assert sha(maskpath)==p['mask_sha256']
        binary=np.asarray(Image.open(maskpath))>0
        assert binary.shape==(680,1200) and int(binary.sum())==p['area']
        rgb=rgb_frame(fid);mask=binary.astype(np.int32)
        full=overlay(rgb,mask,'seed_'+gid+'_full.webp',binary=True)
        yy,xx=np.nonzero(binary);x0=max(0,int(xx.min())-35);x1=min(1200,int(xx.max())+36)
        y0=max(0,int(yy.min())-35);y1=min(680,int(yy.max())+36)
        zoom=overlay(rgb[y0:y1,x0:x1],mask[y0:y1,x0:x1],'seed_'+gid+'_zoom.webp',binary=True)
        original=save_image('seed_'+gid+'_rgb.webp',rgb[y0:y1,x0:x1])
        dest=W/'masks'/(gid+'_'+p['id']+'.png')
        if not dest.exists():dest.symlink_to(maskpath.resolve())
        sem=semantics[gid]
        rep=p['representative']
        items.append(dict(id=gid,mask_id=p['id'],frame=fid,processed=fid//5+1,model=int(rep['model']),
            source_crop=rep.get('crop_key'),source_label=rep.get('label'),source_selection_reason=rep.get('selection_reason'),source_semantic_hit=rep.get('semantic_hit',False),source_semantic_family=rep.get('semantic_family'),source_policy_sam_only=bool(source_groups[gid]['semantic_window_present']),source_group_semantic_families=source_groups[gid]['semantic_families'],area=p['area'],support_count=p['support_count'],
            decision=decision[gid],coverage=coverage.get(gid),semantic={k:sem.get(k) for k in ['top_class','high_excluded_pixels','area']},
            image=zoom,full=full,rgb=original,mask='masks/'+dest.name,mask_sha256=p['mask_sha256'],member_count=g.get('member_count',len(g.get('members',[])))))
    assert {x['id'] for x in items if x['decision']['decision']=='retain'}==set(selected['retained_group_ids'])
    assert set(selected['retained_group_ids'])=={g['id'] for g in kept['groups']}
    data=dict(title='room0 第50帧修复候选',frame_ids=list(FIDS),mapping=mapping,
        timeline=read(R/'mapping_timeline.json'),rois=rois,items=items,stages=selected['stages'],
        retained=selected['retained_group_ids'],input_jobs=len(frontend['jobs']),unique_rgb=len({j['RGB_sha256'] for j in frontend['jobs']}),
        raw_candidates=read(R/'overmerge/web/data.json')['stats'],
        roi_ranked_entries=len(catalog['entries']),unviewable_ROIs=[e['id'] for e in catalog['entries'] if not e['views']],
        pipeline='auto-v3-swap',repair_applied=False,GT_used=False,cutoff=395,
        sourcing_stats=read(R/'semantic_route/semantic.json')['stats'],
        sourcing_margin=read(R/'semantic_route/semantic.json')['config']['margin_threshold'])
    write(W/'data.json',data)
    (W/'data.js').write_text('window.REVIEW_DATA = '+json.dumps(data,ensure_ascii=False)+';\n',encoding='utf-8')
    for name in ['index.html','style.css','app.js']:
        shutil.copyfile(C/'review_web'/name,W/name)
    (W/'downloads').mkdir(exist_ok=True)
    for source,name in [(R/'v2/seed_manifest.json','retained_masks.json'),(R/'v2/selection.json','selection.json'),
                        (R/'mapping_timeline.json','mapping_timeline.json'),(R/'inputs_manifest.json','inputs_manifest.json'),
                        (R/'ply/manifest.json','ply_manifest.json')]:
        shutil.copyfile(source,W/'downloads'/name)
    for p in (R/'ply').glob('*.ply'):
        target=W/'downloads'/p.name
        if not target.exists():target.symlink_to(p)
    with zipfile.ZipFile(W/'downloads/retained_binary_masks.zip','w',zipfile.ZIP_DEFLATED) as archive:
        rows=[]
        for item in items:
            if item['id'] not in data['retained']:continue
            mask=W/item['mask']
            assert sha(mask)==item['mask_sha256']
            archive.write(mask,mask.name)
            rows.append({k:item[k] for k in ['id','mask_id','frame','processed','area','mask_sha256']} | {'file':mask.name})
        archive.writestr('manifest.json',json.dumps(dict(scene='room0',processed_frames=len(FIDS),last_frame=FIDS[-1],masks=rows),ensure_ascii=False,indent=2))
    write(R/'review_complete.json',dict(status='PASS',ROIs=len(rois),seeds=len(items),retained=len(data['retained']),
        assets=len(list(ASSETS.iterdir())),seconds=time.time()-started,data_sha256=sha(W/'data.json'),
        exact_masks_verified=True,source_frames_within_prefix=True,repair_applied=False))
    print('REVIEW_READY',len(items),len(data['retained']),flush=True)

if __name__=='__main__':main()
