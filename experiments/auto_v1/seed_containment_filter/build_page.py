"""Package containment comparisons using saved seed images without rerendering."""
from pathlib import Path
import json,shutil,zipfile
from run import sha,save


def build():
    code=Path(__file__).resolve().parent;cfg=json.loads((code/'config.json').read_text())
    inp,out=Path(cfg['input_directory']),Path(cfg['output_directory'])
    verification=json.loads((out/'verification.json').read_text())
    for name,digest in verification['output_sha256'].items():assert sha(out/name)==digest
    for name,digest in verification['source_sha256'].items():assert sha(name)==digest
    classification=json.loads((out/'classification.json').read_text())
    pairs=json.loads((out/'pair_results.json').read_text())['pairs']
    preview=out/'previews'
    if not preview.exists():preview.symlink_to((inp/'previews').resolve(),target_is_directory=True)
    assert preview.resolve()==(inp/'previews').resolve()
    hashes={};records=classification['records']
    for r in records:
        r['images']={}
        for key in ['thumb','full']:
            name=f'previews/{r["group_id"]}_{key}.jpg'
            r['images'][key]=name;hashes[name]=sha(out/name)
    review=dict(**classification)
    review.update(pairs=[p for p in pairs if p['intersection']>0],
        image_note='Reuse unchanged B seed images; no new rendering or model inference.')
    save(out/'review_data.json',review)
    for name in ['index.html','review.js','review.css']:shutil.copyfile(code/name,out/name)
    verification['page_source_sha256']={name:sha(code/name) for name in ['build_page.py','index.html','review.js','review.css']}
    verification['output_sha256'].update(hashes)
    verification['output_sha256']['review_data.json']=sha(out/'review_data.json')
    save(out/'verification.json',verification)
    package=['pair_results.json','classification.json','selection_binary.json','seed_manifest.json',
             'review_data.json','verification.json','index.html','review.js','review.css',*hashes]
    with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in package:z.write(out/name,name)
    receipt=dict(status='PASS',sha256=sha(out/'review_bundle.zip'),bytes=(out/'review_bundle.zip').stat().st_size,stats=classification['stats'])
    save(out/'delivery_receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':build()
