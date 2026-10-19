"""Render every original source label; four independent CPU workers, no inference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import colorsys
import hashlib
import json
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from run import sha, save

CODE=Path(__file__).resolve().parent
INPUT=Path('/data/chenkejun/CVPR/results/seed_skip_repair_B_20261019')
OUT=Path('/data/chenkejun/CVPR/results/seed45_source_segmentations_20261019')
ORIGINAL=Path('/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4')


def render_source(job):
    source=Path(job['source_path']);rgbp=Path(job['rgb_path']);out=Path(job['output'])
    labels=np.asarray(Image.open(source));rgb=np.asarray(Image.open(rgbp).convert('RGB'))
    assert rgb.shape==(680,1200,3)
    x0,y0,x1,y1=job['box'];assert labels.shape==(y1-y0,x1-x0)
    full=np.zeros((680,1200),dtype=labels.dtype);full[y0:y1,x0:x1]=labels
    all_ids=np.unique(labels[labels>0]);assert len(all_ids)>0
    used=full>0;color=np.full_like(rgb,26)
    for lid in all_ids:
        hue=(int(lid)*.618033988749895+.09)%1
        color[full==lid]=np.rint(np.array(colorsys.hsv_to_rgb(hue,.63,.97))*255).astype(np.uint8)
    overlay=rgb.copy();overlay[used]=np.rint(rgb[used]*.47+color[used]*.53).astype(np.uint8)
    edge=np.zeros(full.shape,bool);edge[:,1:]|=full[:,1:]!=full[:,:-1];edge[1:]|=full[1:]!=full[:-1];edge&=used
    overlay[edge]=[250,250,250]
    assert np.array_equal(np.unique(full[full>0]),all_ids)
    for s in job['seed_records']:
        mask_path=ORIGINAL/s['mask'];assert sha(mask_path)==s['mask_sha256']
        assert np.array_equal(full==s['source_label'],np.asarray(Image.open(mask_path))>0)
    iscrop=job['box']!=[0,0,1200,680]
    overlay_im=Image.fromarray(overlay);mask_im=Image.fromarray(color)
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),13) if fontpath.exists() else ImageFont.load_default()
    # Every label keeps its pixels; text is a display aid only.
    for canvas in [overlay_im,mask_im]:
        draw=ImageDraw.Draw(canvas)
        if iscrop:draw.rectangle((x0,y0,x1-1,y1-1),outline=(255,217,80),width=3)
        for lid in all_ids:
            yy,xx=np.nonzero(full==lid)
            if len(xx)<40:continue
            k=int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2));x,y=int(xx[k]),int(yy[k]);label=str(int(lid))
            box=draw.textbbox((x,y),label,font=font);draw.rectangle((box[0]-2,box[1]-1,box[2]+2,box[3]+1),fill=(23,31,28))
            draw.text((x,y),label,font=font,fill=(255,255,255))
    sid=job['source_id'];names={k:f'assets/{sid}_{k}.{ext}' for k,ext in [('overlay','jpg'),('mask','png'),('rgb','jpg'),('labels','png')]}
    overlay_im.save(out/names['overlay'],quality=94);mask_im.save(out/names['mask'],compress_level=3)
    shutil.copyfile(rgbp,out/names['rgb']);shutil.copyfile(source,out/names['labels'])
    return dict(source_id=sid,source_path=str(source),source_sha256=sha(source),rgb_path=str(rgbp),rgb_sha256=sha(rgbp),
        frame=job['frame'],box=job['box'],crop=iscrop,model_name=job['model_name'],mask_count=len(all_ids),
        label_ids=all_ids.astype(int).tolist(),groups=job['groups'],images=names,
        output_sha256={name:sha(out/name) for name in names.values()})


def main():
    assert sha(INPUT/'seed_manifest.json')=='15ee80b2b200304b2555937fee29edd88eb532861fe98f5adeab124d67cc1658'
    manifest=json.loads((INPUT/'seed_manifest.json').read_text());assert len(manifest['groups'])==45
    cam_path=Path('/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room2/configs/raw.json')
    camera=json.loads(cam_path.read_text());scene=Path(camera['source']['scene_root'])
    (OUT/'assets').mkdir(parents=True,exist_ok=True)
    jobs={};records=[];modelnames={0:'原图 CropFormer',1:'局部 CropFormer',2:'SAM2.1'}
    for g in manifest['groups']:
        s=g['primary'];r=s['representative'];key=r['source_path'];sid='S'+hashlib.sha256(key.encode()).hexdigest()[:12]
        seed=dict(group_id=g['id'],frame=s['frame'],source_id=sid,source_label=r['label'],mask=s['mask'],mask_sha256=s['mask_sha256'])
        records.append(seed)
        if key not in jobs:
            jobs[key]=dict(source_id=sid,source_path=key,box=r['source_box'],frame=s['frame'],model_name=modelnames[r['model']],
                rgb_path=str(scene/camera['source']['rgb_pattern'].format(frame=s['frame'])),output=str(OUT),groups=[],seed_records=[])
        assert jobs[key]['box']==r['source_box'] and jobs[key]['frame']==s['frame']
        jobs[key]['groups'].append(g['id']);jobs[key]['seed_records'].append(seed)
    assert len(jobs)==41
    with ProcessPoolExecutor(max_workers=4) as pool:
        sources=list(pool.map(render_source,jobs.values()))
    source_index={s['source_id']:s for s in sources}
    data=dict(records=records,sources=source_index,stats=dict(seeds=45,source_outputs=41,frames=34))
    save(OUT/'review_data.json',data)
    html=(CODE/'fullseg.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    save(OUT/'verification.json',dict(status='PASS',input_manifest_sha256=sha(INPUT/'seed_manifest.json'),camera_config_sha256=sha(cam_path),
        source_sha256={p:h for s in sources for p,h in [(s['source_path'],s['source_sha256']),(s['rgb_path'],s['rgb_sha256'])]},
        code_sha256={name:sha(CODE/name) for name in ['build_fullseg.py','fullseg.html']},
        checks=['All 45 seed masks exactly match their original source label, with original coordinates',
                'All positive labels of all 41 source outputs rendered without filtering',
                'Cropped outputs pasted into full 1200x680 frames, original inference box marked',
                'Original source label PNGs copied byte-for-byte; original RGB files copied byte-for-byte',
                'No model inference, mask edits, seed selection changes, E1 or GT use'],
        output_sha256={name:d for s in sources for name,d in s['output_sha256'].items()}))
    for s in sources:
        assert sha(s['source_path'])==s['source_sha256'] and sha(s['rgb_path'])==s['rgb_sha256']
    from update_fullseg_view import update
    update(OUT,INPUT,package=True)
    print((OUT/'delivery_receipt.json').read_text())


if __name__=='__main__':main()
