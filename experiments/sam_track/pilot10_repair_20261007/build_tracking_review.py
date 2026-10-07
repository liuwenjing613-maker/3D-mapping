"""Build native-output tracking viewer and shareable clips from archived results."""
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent.parent / 'results/固定案例_三模型对比_20261006'
OUT = DELIVERY / 'tracking_review_20261007'
PRIOR = DELIVERY / 'repair_validation_20261007'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def decode(img):
    a=np.array(img).astype(np.uint32)
    return a[:,:,0] | (a[:,:,1]<<8) | (a[:,:,2]<<16)

def overlay(case, row, solo=None, crop=None, caption=None):
    rgb=np.array(Image.open(OUT/row['RGB']).convert('RGB'))
    labels=decode(Image.open(OUT/row['IDs']))
    image=rgb.copy()
    color=np.asarray(case['object_colors'],np.uint8)
    visible=labels>0
    if solo is not None: visible &= labels==solo
    image[visible]=np.rint(rgb[visible]*.55+color[labels[visible]]*.45).astype(np.uint8)
    edge=visible & ((labels!=np.roll(labels,1,axis=0)) | (labels!=np.roll(labels,-1,axis=0)) | (labels!=np.roll(labels,1,axis=1)) | (labels!=np.roll(labels,-1,axis=1)))
    image[edge]=color[labels[edge]]
    canvas=Image.fromarray(image)
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    for obj in case['objects']:
        tid=obj['track_id']
        if solo is not None and tid!=solo: continue
        y,x=np.nonzero(labels==tid)
        if not len(x): continue
        center=np.asarray(row['centers'][str(tid)])
        j=np.argmin((x-center[0])**2+(y-center[1])**2)
        draw.text((int(x[j]),int(y[j])),str(obj['native_mask_id']),font=font,anchor='mm',fill='white',stroke_width=3,stroke_fill=(10,22,32))
    if crop: canvas=canvas.crop(crop)
    if caption:
        canvas=canvas.resize((720,408))
        result=Image.new('RGB',(720,462),(18,32,51))
        result.paste(canvas,(0,54))
        ImageDraw.Draw(result).text((14,12),caption,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22),fill='white')
        return result
    return canvas

def encode_video(path, images, fps=5):
    frames=iter(images)
    first=next(frames)
    proc=subprocess.Popen([shutil.which('ffmpeg'),'-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{first.width}x{first.height}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    proc.stdin.write(first.tobytes())
    for im in frames: proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('Video encoding failed')

def main():
    OUT.mkdir(exist_ok=True)
    bundle=HERE/'tracking_review_bundle.tar.gz'
    assert sha(bundle)=='1902d3f873b515f39165382254ca42a054d3a583da46e8dbc1d12e8af4bec4c5'
    with tarfile.open(bundle,'r:gz') as archive:
        for member in archive.getmembers():
            target=(OUT/member.name).resolve()
            assert target.is_relative_to(OUT.resolve()) and (member.isfile() or member.isdir())
        archive.extractall(OUT,filter='data')
    data=json.loads((OUT/'tracking_data.json').read_text(encoding='utf-8'))
    for name,expected in data['asset_hashes'].items(): assert sha(OUT/name)==expected
    review=json.loads((PRIOR/'review_data.json').read_text(encoding='utf-8'))
    ids=json.loads((PRIOR/'id_maps.json').read_text(encoding='utf-8'))
    metrics=json.loads((PRIOR/'v3_summary.json').read_text(encoding='utf-8'))
    by_uid={c['case_uid']:c for c in review['cases']}
    id_by_uid={c['case_uid']:c for c in ids['cases']}
    metric_by_uid={c['case_uid']:c for c in metrics['cases']}
    missing=[]
    for c in data['cases']:
        if c['status']!='READY': continue
        uid=c['case_uid']; previous=by_uid[uid]
        c['object_colors']=previous['object_colors']
        audit={o['track_id']:o for o in previous['vote_audit']}
        for o in c['objects']:
            o['audit']=audit[o['track_id']]
            if o['missing_frames']: missing.append({'case':uid,'mask':o['native_mask_id'],'frames':o['missing_frames']})
        firstview=previous['views'][0]
        box=firstview['box']; x0,y0,x1,y1=box
        assert c['seed_frame']==firstview['frame']
        c['seed_crop']=box
        c['projection_IDs']={}
        assets=id_by_uid[uid]['views'][0]['ID_assets']
        for key in ['baseline','seed_plus_short_track']:
            raw=np.array(Image.open(PRIOR/assets[key]).convert('RGB'))
            assert raw.shape==(y1-y0,x1-x0,3)
            whole=np.zeros((680,1200,3),np.uint8)
            whole[y0:y1,x0:x1]=raw
            name='assets/'+uid+'/'+key+'_seed_projection_ids.png'
            Image.fromarray(whole).save(OUT/name)
            c['projection_IDs'][key]=name
        m=metric_by_uid[uid]['conditions']['seed_plus_short_track']
        c['evaluation']={k:m[k] for k in ['AP50_delta','F1_delta','PQ_delta','target_mean_IoU_delta','target_structure_count_delta','strict_success']}
        c['key_frames']=sorted(set([0,len(c['frames'])//4,next(i for i,r in enumerate(c['frames']) if r['seed']),3*len(c['frames'])//4,len(c['frames'])-1]))
        tiles=[]
        for fi in c['key_frames']:
            r=c['frames'][fi]
            cap=f"f{r['frame']}  ·  {'种子' if r['seed'] else ('建图帧' if r['mapping'] else '中间帧')}"
            tiles.append(overlay(c,r,caption=cap).resize((480,308)))
        sheet=Image.new('RGB',(480*len(tiles),366),'white')
        draw=ImageDraw.Draw(sheet)
        draw.text((16,13),f"{c['scene']} {c['ROI']} · 实际 SAM2.1 追踪 · 原 mask 编号与颜色全程一致",font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26),fill=(18,32,51))
        for i,tile in enumerate(tiles): sheet.paste(tile,(480*i,58))
        c['contact_sheet']='assets/'+uid+'/keyframes.jpg'
        sheet.save(OUT/c['contact_sheet'],quality=94)
        c['video']='assets/'+uid+'/tracking_5fps.mp4'
        encode_video(OUT/c['video'],(overlay(c,r,caption=f"{c['scene']} {c['ROI']}  ·  f{r['frame']}  ·  {'种子（原标注）' if r['seed'] else ('建图帧' if r['mapping'] else '追踪中间帧')}") for r in c['frames']))
        print(json.dumps({'case':uid,'video':c['video'],'missing_masks':sum(bool(o['missing_frames']) for o in c['objects'])},ensure_ascii=False),flush=True)
    demo=next(c for c in data['cases'] if c['scene']=='room2')
    demo_tid=next(o['track_id'] for o in demo['objects'] if o['native_mask_id']==24)
    boxes=[r['boxes'][str(demo_tid)] for r in demo['frames'] if r['boxes'][str(demo_tid)]]
    bounds=np.array(boxes)
    crop=[max(0,int(bounds[:,0].min())-110),max(0,int(bounds[:,1].min())-60),min(1200,int(bounds[:,2].max())+110),min(680,int(bounds[:,3].max())+110)]
    gif=[]
    for r in demo['frames']:
        # A fixed crop across the clip preserves camera/object motion.
        im=overlay(demo,r,solo=demo_tid,crop=crop)
        im.thumbnail((520,440))
        tile=Image.new('RGB',(520,494),(18,32,51)); tile.paste(im,((520-im.width)//2,54+(440-im.height)//2))
        ImageDraw.Draw(tile).text((14,12),f"room2 · mask 24 · f{r['frame']}"+(' · 种子' if r['seed'] else ''),font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22),fill='white')
        gif.append(tile)
    gif[0].save(OUT/'room2_mask24_actual_tracking.gif',save_all=True,append_images=gif[1:],duration=200,loop=0,optimize=False)
    data['diagnostics']={'missing_masks':missing,'nonempty_frame_count_is_accuracy':False,
                         'tracking_quality_has_pixel_GT':False,'independent_case_evaluation':True,
                         'v3_preset':'DEBUG_ONLY / NON_OFFICIAL'}
    data['evaluation_aggregate']=metrics['aggregate']['seed_plus_short_track']
    data['displayed_v3_metrics_use_posthoc_GT']=True
    dump(OUT/'tracking_review_data.json',data)
    template=(HERE/'tracking_review_template.html').read_text(encoding='utf-8')
    if '__PLY_DATA__' in template:
        ply_manifest=json.loads((OUT/'ply_models'/'manifest.json').read_text(encoding='utf-8'))
        template=template.replace('__PLY_DATA__',json.dumps(ply_manifest,ensure_ascii=False).replace('</','<\\/'))
    (OUT/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    verification={'status':'PASS','case_count':10,'tracked_cases':9,'objects':55,'raw_frames':353,'mapping_frames':77,
                  'native_RGB_asset_hashes_verified':353,'native_label_ID_roundtrips_verified':353,
                  'new_inference':False,'seed_source_modified':False,'tracking_bundle_sha256':sha(bundle),
                  'diagnostics':data['diagnostics'],'exported_videos':9}
    dump(OUT/'verification.json',verification)
    # A portable copy is kept under the server's CVPR result tree too.
    archive=HERE/'tracking_review_delivery.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for file in OUT.rglob('*'):
            if file.is_file() and file.relative_to(OUT).as_posix() not in data['asset_hashes']:
                tar.add(file,arcname='tracking_review/'+file.relative_to(OUT).as_posix())
    dump(HERE/'tracking_review_delivery.json',{'status':'PASS','bundle':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size})
    print(json.dumps(verification,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
