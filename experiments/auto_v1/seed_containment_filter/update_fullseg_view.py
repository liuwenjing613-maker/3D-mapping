"""Add a seed-only switch by reusing B's saved images; never rerender masks."""
from pathlib import Path
import argparse,json,shutil,zipfile
from run import sha,save


def update(out,inp,package=False):
    code=Path(__file__).resolve().parent
    assert sha(inp/'selection_binary.json')=='5fd2a22f94b2812b246b1ce14b45e759b70e9ad295c1c421c90502310bbbd2f1'
    selection=json.loads((inp/'selection_binary.json').read_text(encoding='utf-8'))
    data=json.loads((out/'review_data.json').read_text(encoding='utf-8'))
    assert {r['group_id'] for r in data['records']}=={r['group_id'] for r in selection['retained']}
    (out/'seed_masks').mkdir(exist_ok=True)
    hashes={}
    for r in data['records']:
        name=f'seed_masks/{r["group_id"]}_full.jpg'
        source=inp/'previews'/Path(name).name
        if not (out/name).exists():shutil.copyfile(source,out/name)
        assert sha(source)==sha(out/name)
        hashes[name]=sha(source);r['seed_image']=name
    save(out/'review_data.json',data)
    html=(code/'fullseg.html').read_text(encoding='utf-8').replace('__DATA__',json.dumps(data,ensure_ascii=False))
    (out/'index.html').write_text(html,encoding='utf-8')
    verification=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    verification['output_sha256'].update(hashes)
    verification['seed_image_source']='Unchanged B full-frame seed overlays, byte-for-byte copies'
    verification['code_sha256'].update({n:sha(code/n) for n in ['fullseg.html','update_fullseg_view.py']})
    save(out/'verification.json',verification)
    if package:
        with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['index.html','review_data.json','verification.json',*verification['output_sha256']]:z.write(out/name,name)
        save(out/'delivery_receipt.json',dict(status='PASS',sha256=sha(out/'review_bundle.zip'),bytes=(out/'review_bundle.zip').stat().st_size,stats=data['stats']))
    print('Added seed-only switch; 45 cached images verified; no new mask rendering.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('/data/chenkejun/CVPR/results/seed45_source_segmentations_20261019'))
    p.add_argument('--input',type=Path,default=Path('/data/chenkejun/CVPR/results/seed_skip_repair_B_20261019'));p.add_argument('--package',action='store_true')
    a=p.parse_args();update(a.output,a.input,a.package)
