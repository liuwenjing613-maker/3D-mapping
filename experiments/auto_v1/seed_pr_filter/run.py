"""Freeze the 42-seed p/r filter and package a page referring to existing images."""
from pathlib import Path
import copy,hashlib,json,shutil,zipfile
from rules import excluded


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    code=Path(__file__).resolve().parent;cfg=json.loads((code/'config.json').read_text())
    inp=Path(cfg['input_directory']);base=Path(cfg['coverage_directory']);source=Path(cfg['source_gallery_directory']);out=Path(cfg['output_directory'])
    assert not (out/'classification.json').exists(),'Refuse to overwrite frozen selection.'
    locked={}
    def lock(path,expected=None):
        digest=sha(path)
        if expected is not None:assert digest==expected,('Changed input',str(path))
        locked[str(path)]=digest
    for path,digest in [(inp/'selection_binary.json',cfg['input_selection_sha256']),
                        (inp/'seed_manifest.json',cfg['input_manifest_sha256']),
                        (base/'selection_binary.json',cfg['coverage_selection_sha256']),
                        (base/'review_data.json',cfg['coverage_review_sha256']),
                        (source/'review_data.json',cfg['source_gallery_review_sha256'])]:lock(path,digest)
    selected=json.loads((inp/'selection_binary.json').read_text());manifest=json.loads((inp/'seed_manifest.json').read_text())
    coverage=json.loads((base/'review_data.json').read_text());sources=json.loads((source/'review_data.json').read_text())
    previous={r['group_id']:r for r in selected['retained']};groups={g['id']:g for g in manifest['groups']}
    b={r['group_id']:r for r in coverage['records']};s={r['group_id']:r for r in sources['records']}
    assert len(previous)==cfg['input_count']==42 and set(previous)==set(groups)
    assert not set(previous)&{r['group_id'] for r in selected['excluded']}
    image_cache={}
    def image_url(directory,relative,web_root):
        path=directory/relative;url=web_root+relative
        if url not in image_cache:
            lock(path);image_cache[url]=dict(path=str(path),sha256=locked[str(path)])
        return url
    records=[]
    for gid in sorted(previous):
        rec=copy.deepcopy(b[gid]);seed=groups[gid]['primary'];origin=s[gid];src=sources['sources'][origin['source_id']]
        assert rec['decision']=='retain'
        for key in ['mask_sha256','frame','original_track_id']:assert rec[key]==previous[gid][key]
        assert rec['mask_sha256']==origin['mask_sha256']==seed['mask_sha256']
        assert rec['frame']==origin['frame']==src['frame']==seed['frame']
        assert rec['support_count']==seed['support_count']
        lock(inp/seed['mask'],seed['mask_sha256'])
        c,n,i=rec['intersection'],rec['support_count'],rec['instance_visible_count']
        assert rec['p']==c/n and rec['r']==(c/i if i else 0.)
        remove=excluded(c,n,i)
        rec.update(previous_B_decision=rec['decision'],previous_containment_decision='retain',
            decision='exclude' if remove else 'retain',
            p_above_10=bool(10*c>n),r_below_20=bool(i>0 and 5*c<i),
            reason='p_gt_10_and_r_lt_20' if remove else ('p_not_above_10' if 10*c<=n else 'r_not_below_20'),
            source_label=origin['source_label'],source_model=src['model_name'],source_mask_count=src['mask_count'])
        rec['images']={k:image_url(base,v,cfg['coverage_web_root']) for k,v in rec['images'].items()}
        for key in ['overlay','mask','rgb']:
            rec['images'][key]=image_url(source,src['images'][key],cfg['source_gallery_web_root'])
        records.append(rec)
    removed=[r for r in records if r['decision']=='exclude'];kept=[r for r in records if r['decision']=='retain']
    assert {r['group_id'] for r in removed}.isdisjoint(r['group_id'] for r in kept)
    assert {r['group_id'] for r in records}==set(previous)
    for path,digest in locked.items():assert sha(path)==digest
    out.mkdir(parents=True,exist_ok=True)
    stats=dict(input=42,excluded=len(removed),retained=len(kept),source_frames=len({r['frame'] for r in records}))
    payload=dict(schema='p-gt10-r-lt20-trial-v1',config=cfg,stats=stats,records=records,
                 prior_containment_exclusions=[r['group_id'] for r in selected['excluded']])
    save(out/'classification.json',payload)
    save(out/'selection_binary.json',dict(schema='p-gt10-r-lt20-selection-v1',counts=dict(excluded=len(removed),retained=len(kept)),
        input_selection_sha256=cfg['input_selection_sha256'],excluded=removed,retained=kept,tracking_requested=False,map_write=False))
    new_manifest=copy.deepcopy(manifest)
    new_manifest.update(groups=[groups[r['group_id']] for r in kept],pr_filter_config=cfg,
        pr_filter_excluded_group_ids=[r['group_id'] for r in removed],pr_filter_input_manifest_sha256=cfg['input_manifest_sha256'],
        status='p_gt10_r_lt20_screening_only',tracking_requested=False,map_write=False)
    save(out/'seed_manifest.json',new_manifest)
    (out/'masks').symlink_to((inp/'masks').resolve(),target_is_directory=True)
    for r in kept:assert sha(out/groups[r['group_id']]['primary']['mask'])==r['mask_sha256']
    save(out/'review_data.json',payload)
    save(out/'cached_images.json',image_cache)
    for name in ['index.html','review.js','review.css']:shutil.copyfile(code/name,out/name)
    save(out/'verification.json',dict(status='PASS',source_sha256=locked,
        code_sha256={p.name:sha(p) for p in code.iterdir() if p.suffix in ['.py','.json','.html','.css','.js']},
        checks=['User-confirmed 42-seed scope; earlier 3 exclusions not reintroduced',
                'p/r reused exactly from frozen B strict-one-vote unrepaired map measurements',
                'Integer strict p>10% AND r<20%; p==10% and r==20% retained',
                'Decisions cover all 42 inputs exactly once, original masks and IDs unchanged',
                'All image URLs refer to saved, hashed images; no segmentation or image rerendering',
                'No E1 run, map writes, GT use or evaluation changes'],
        output_sha256={n:sha(out/n) for n in ['classification.json','selection_binary.json','seed_manifest.json','review_data.json','cached_images.json','index.html','review.js','review.css']}))
    names=['classification.json','selection_binary.json','seed_manifest.json','review_data.json','cached_images.json','verification.json','index.html','review.js','review.css']
    with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(out/name,name)
    receipt=dict(status='PASS',sha256=sha(out/'review_bundle.zip'),bytes=(out/'review_bundle.zip').stat().st_size,stats=stats)
    save(out/'delivery_receipt.json',receipt);print(json.dumps(receipt));print('EXCLUDED:',','.join(r['group_id'] for r in removed))


if __name__=='__main__':main()
