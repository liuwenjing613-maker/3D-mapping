"""Preserve and validate the new human export against the exact selection assets."""
from pathlib import Path
import base64, hashlib, io, json, shutil
import numpy as np
from PIL import Image

WORK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAGE = WORK / 'results/固定案例_三模型对比_20261006/room2_selection_20261007'
SOURCE = WORK / 'room2_修复候选选择_20261007.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def decode(image):
    a = np.asarray(image, np.uint32)
    return a if a.ndim == 2 else a[..., 0] | (a[..., 1] << 8) | (a[..., 2] << 16)

def main():
    selection = json.loads(SOURCE.read_text(encoding='utf8'))
    view = json.loads((PAGE / 'viewer_data.json').read_text(encoding='utf8'))
    assert selection['schema'] == 'cvpr_roi_human_selection_v1'
    assert selection['scene_scope'] == 'room2' and not selection['is_qa']
    assert len(selection['selections']) == selection['total_fixed_cases'] == 5
    assert set(selection['pilot_case_uids']) == {x['case_uid'] for x in selection['selections']}
    target = OUT / 'inputs'
    target.mkdir(exist_ok=True)
    shutil.copyfile(SOURCE, target / 'human_selection.json')
    rows, oid = [], 1
    for choice in selection['selections']:
        assert choice['scene'] == 'room2' and not choice.get('is_qa', False)
        if choice['choice'] == 'unreliable':
            assert not choice['mask_ids'] and choice['label_asset'] is None
            rows.append({'status': 'NO_RELIABLE_SEED', 'case_uid': choice['case_uid'], 'source_choice': choice})
            continue
        crop = view['crops'][choice['crop_key']]
        assert crop['box'] == choice['crop_xyxy'] and crop['wh'] == choice['native_crop_wh']
        mi = ['original','cropformer_local','sam2_1_local'].index(choice['choice'])
        assert mi == choice['model_index']
        model = crop['models'][mi]
        asset = (PAGE / model['ids_file']).resolve()
        assert asset == (PAGE.parent / choice['label_asset']).resolve()
        labels = decode(Image.open(asset))
        np.testing.assert_array_equal(labels, decode(Image.open(io.BytesIO(base64.b64decode(model['ids'].split(',',1)[1])))))
        assert list(Image.open(asset).size) == crop['wh']
        assert len(set(choice['mask_ids'])) == len(choice['mask_ids'])
        objects = []
        seed = np.zeros((680,1200), np.uint16)
        x0,y0,x1,y1 = choice['crop_xyxy']
        for mid in sorted(choice['mask_ids']):
            mask = labels == mid
            assert mask.any(), (choice['case_uid'], mid)
            seed[y0:y1,x0:x1][mask] = oid
            objects.append({'track_id': oid, 'native_mask_id': mid, 'crop_pixels': int(mask.sum()),
                            'touches_crop_border': bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())})
            oid += 1
        name = choice['case_uid'] + '_selected_seed.png'
        Image.fromarray(seed).save(target / name)
        shutil.copyfile(asset, target / (choice['case_uid'] + '_asset.png'))
        rows.append({'status':'READY','case_uid':choice['case_uid'],'scene':'room2','source_choice':choice,
                     'objects':objects,'seed_file':name,'seed_sha256':sha(target/name),
                     'source_label_sha256':sha(asset),'source_label_local_path':str(asset),
                     'selection_viewer_sha256':sha(PAGE/'viewer_data.json')})
    report = {'status':'PASS','annotation_sha256':sha(SOURCE),'annotation_exported_at':selection['exported_at'],
              'scene':'room2','total_cases':len(rows),'repair_cases':sum(r['status']=='READY' for r in rows),
              'selected_masks':oid-1,'seeds':rows,'GT_used':False,
              'original_full_masks_to_be_restored_and_verified_against_raw_server_source':True}
    (target/'local_ingest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='seeds'},ensure_ascii=True))

if __name__ == '__main__': main()
