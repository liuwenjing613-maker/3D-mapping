"""Build a two-category review with immutable pair evidence and comparison-only manual labels."""
from pathlib import Path
import json, shutil, zipfile
import numpy as np
from PIL import Image, ImageDraw
from run import sha


def main():
    code=Path(__file__).resolve().parent
    cfg=json.loads((code/'config.json').read_text());out=Path(cfg['output_directory']);inp=Path(cfg['input_directory'])
    done=json.loads((out/'complete.json').read_text());assert done['status']=='PASS'
    verification=json.loads((out/'verification.json').read_text())
    for name,digest in verification['output_sha256'].items():assert sha(out/name)==digest
    data=json.loads((out/'pair_results.json').read_text())
    original=Path(cfg['original_directory']);manifest=json.loads((inp/'seed_manifest.json').read_text())
    seeds={g['id']:g['primary'] for g in manifest['groups']}
    with np.load(original/'surface_supports.npz') as z:ids=z['indices'];offsets=z['offsets']
    supports={g:ids[offsets[s['index']]:offsets[s['index']+1]] for g,s in seeds.items()}
    camcfg=json.loads(Path(cfg['camera_config']).read_text());source=camcfg['source'];scene=Path(source['scene_root'])
    preview=out/'previews';preview.mkdir(exist_ok=True)
    # Existing context previews use the unchanged mask and original RGB; copy those, not new model inference.
    semantics=json.loads((inp/'binary_comparison.json').read_text());sem={r['group_id']:r for r in semantics['records']}
    rgb_cache={};visibility={}
    def rgb(frame):
        if frame not in rgb_cache:rgb_cache[frame]=np.asarray(Image.open(scene/source['rgb_pattern'].format(frame=frame)).convert('RGB'))
        return rgb_cache[frame]
    def vis(frame):
        if frame not in visibility:
            with np.load(Path(cfg['visibility_directory'])/f'frame{frame:06d}_visible.npz') as z:visibility[frame]=(z['vertex_index'],z['uv'])
        return visibility[frame]
    for rec in data['records']:
        g=rec['group_id'];s=seeds[g]
        rec['images']={}
        for key in ['context','masked']:
            name=sem[g]['images'][key];shutil.copyfile(inp/name,out/name);rec['images'][key]=name
        a=np.asarray(Image.open(inp/s['mask']))>0
        arr=rgb(s['frame']).copy();arr[a]=(arr[a]*.55+np.array([42,209,166])*.45).astype(np.uint8)
        full=Image.fromarray(arr);full.thumbnail((1200,680))
        name=f'previews/{g}_full.jpg';full.save(out/name,quality=90);rec['images']['full']=name
    # Draw every nontrivial positive-overlap pair; remaining comparisons can still inspect both exact source masks.
    rendered=0
    for p in data['pairs']:
        if p['score']<.1 or p['intersection']<20:continue
        a,b=supports[p['a']],supports[p['b']]
        va,_=vis(p['frame_a']);vb,_=vis(p['frame_b'])
        av=np.intersect1d(a,vb,assume_unique=True);bv=np.intersect1d(b,va,assume_unique=True)
        shared=np.intersect1d(av,bv,assume_unique=True)
        only_a=np.setdiff1d(av,bv,assume_unique=True);only_b=np.setdiff1d(bv,av,assume_unique=True)
        assert len(shared)==p['intersection'] and len(av)==p['visible_a'] and len(bv)==p['visible_b']
        p['images']={}
        for side,g in [('a',p['a']),('b',p['b'])]:
            s=seeds[g];frame=s['frame'];vid,uv=vis(frame)
            arr=(rgb(frame)*.65).astype(np.uint8)
            canvas=Image.fromarray(arr);draw=ImageDraw.Draw(canvas)
            drawn=[]
            for points,color in [(only_a,(255,169,59)),(only_b,(88,169,255)),(shared,(48,242,163))]:
                if not len(points):continue
                ii=np.searchsorted(vid,points);assert np.array_equal(vid[ii],points)
                xy=uv[ii];drawn.append(xy)
                for x,y in xy:draw.ellipse((int(x)-1,int(y)-1,int(x)+1,int(y)+1),fill=color)
            x0,y0,x1,y1=s['bbox']
            if drawn:
                xy=np.concatenate(drawn);x0=min(x0,int(xy[:,0].min()));x1=max(x1,int(xy[:,0].max())+1)
                y0=min(y0,int(xy[:,1].min()));y1=max(y1,int(xy[:,1].max())+1)
            box=(max(0,x0-45),max(0,y0-45),min(1200,x1+45),min(680,y1+45))
            canvas=canvas.crop(box);canvas.thumbnail((800,600))
            name=f'previews/{p["a"]}_{p["b"]}_{side}_points.jpg';canvas.save(out/name,quality=92)
            p['images'][side]=name
        rendered+=1
    # Manual decisions are read only after automatic outputs were frozen; no pair identity labels are inferred.
    human_path=inp/'human_reference.json';human=json.loads(human_path.read_text())
    assert human['status']=='confirmed' and not human['qa_only']
    assert sha(human_path)=='e98f55e00b1f0f667a954b6c8ef540269edbdc900f9abba4cdb45318b260301c'
    manual={r['group_id']:('retain',r) for r in human['included']}
    manual.update({r['group_id']:('exclude',r) for r in human['excluded']})
    counts={k:0 for k in ['both_keep','both_exclude','automatic_only_exclude','manual_only_exclude']}
    for r in data['records']:
        h,ref=manual[r['group_id']]
        for k in ['mask_id','mask_sha256','frame','original_track_id']:assert r[k]==ref[k]
        r['human_decision']=h
        key=('both_keep' if h=='retain' else 'manual_only_exclude') if r['dedup_decision']=='retain' else ('both_exclude' if h=='exclude' else 'automatic_only_exclude')
        counts[key]+=1
    data['manual_comparison']=dict(counts=counts,revision=human['revision'],reference_sha256=sha(human_path),
        note='人工清单没有逐对实例对应和排除原因；数量一致不能当作去重准确率，人工排除但程序保留也不全是漏去重。')
    data['verification']=verification;data['rendered_pairs']=rendered
    data['input_pair_results_sha256']=sha(out/'pair_results.json')
    (out/'review_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ['index.html','review.js','review.css']:shutil.copyfile(code/name,out/name)
    (out/'comparison.json').write_text(json.dumps(data['manual_comparison'],ensure_ascii=False,indent=2),encoding='utf-8')
    package=['review_data.json','comparison.json','pair_results.json','selection_binary.json','seed_manifest.json','input_freeze.json','verification.json','complete.json','index.html','review.js','review.css']
    with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in package:z.write(out/name,name)
        for p in sorted(preview.glob('*.jpg')):z.write(p,p.relative_to(out).as_posix())
    receipt=dict(sha256=sha(out/'review_bundle.zip'),bytes=(out/'review_bundle.zip').stat().st_size,stats=data['stats'],rendered_pairs=rendered)
    (out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt));print('Manual selection comparison only:',counts)


if __name__=='__main__':main()
