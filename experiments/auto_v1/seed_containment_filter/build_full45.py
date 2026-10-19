"""Independent whole-source-frame gallery, fixed to B's 45 retained seeds."""
from pathlib import Path
import argparse
import hashlib
import json
from PIL import Image


def build(inp, out, image_prefix):
    selection_path = inp / 'selection_binary.json'
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == '5fd2a22f94b2812b246b1ce14b45e759b70e9ad295c1c421c90502310bbbd2f1'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    assert len(selection['retained']) == 45
    records, image_hashes = [], {}
    for r in selection['retained']:
        path = inp / 'previews' / f'{r["group_id"]}_full.jpg'
        with Image.open(path) as im:
            assert im.size == (1200, 680)
        image_hashes[r['group_id']] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({k:r[k] for k in ['group_id','mask_id','frame','area','support_count','mask_sha256']})
        records[-1]['image'] = image_prefix + path.name
    data = dict(records=records, input_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())
    html = (Path(__file__).resolve().parent / 'full45.html').read_text(encoding='utf-8').replace('__SEED_DATA__', json.dumps(data, ensure_ascii=False))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'index.html').write_text(html, encoding='utf-8')
    (out / 'verification.json').write_text(json.dumps(dict(status='PASS', input_count=45, full_frame_size=[1200,680], image_sha256=image_hashes, input_selection_sha256=data['input_selection_sha256']), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Built 45 original full-frame Mask views:', str(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('/data/chenkejun/CVPR/results/seed_skip_repair_B_20261019'))
    p.add_argument('--output', type=Path, default=Path('/data/chenkejun/CVPR/results/seed45_full_masks_20261019'))
    p.add_argument('--image-prefix', default='../seed_skip_repair_B_20261019/previews/')
    a = p.parse_args()
    build(a.input, a.output, a.image_prefix)
