from pathlib import Path
import json, shutil, hashlib

H = Path(__file__).resolve().parent
ROOT = H.parents[1] / 'results/固定案例_三模型对比_20261006'
RAW = ROOT / 'room2_E1_prerepair_CUT_top20_20261015'
OUT = ROOT / 'room2_E1_instance_seed_groups_20261015'
OUT.mkdir(exist_ok=True)
data = json.loads((H/'grouping.json').read_text(encoding='utf-8'))
raw = json.loads((RAW/'viewer_data.json').read_text(encoding='utf-8'))
frames = {}
roi_names = {}
for c in raw['cases']:
    roi_names[c['uid']] = c['aliases']
    for v in c['views']:
        assert (RAW/v['full_rgb']).is_file()
        frames.setdefault(v['frame'], '../'+RAW.name+'/'+v['full_rgb'])
for s in data['seeds']:
    a = s['representative']
    s['rgb'] = frames[s['frame']]
    s['color'] = raw['crops'][a['crop_key']]['models'][a['model']]['colors'][str(a['label'])]
    s['roi_names'] = sorted({name for uid in s['rois'] for name in roi_names[uid]})
    s['source_link'] = '../room2_E1_selected_ROI_20261015_r3/index.html?roi='+a['uid']+'&crop='+a['crop_key']
    assert hashlib.sha256((H/s['mask']).read_bytes()).hexdigest() == s['mask_sha256']
for g in data['groups']:
    g['roi_names'] = sorted({name for j in g['members'] for name in data['seeds'][j]['roi_names']})
data['display'] = dict(total_roi=len(raw['cases']), nonempty_roi=data['stats']['rois'],
    grouping_sha256=hashlib.sha256((H/'grouping.json').read_bytes()).hexdigest(),
    note='固定主种子分组；每组 TSDF 点数最多的一张是唯一主种子。其他成员保留用于检查。')
(OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
shutil.copytree(H/'masks',OUT/'masks',dirs_exist_ok=True)
valid_masks = {s['mask'] for s in data['seeds']}
for p in (OUT/'masks').glob('*.png'):
    assert p.resolve().parent == (OUT/'masks').resolve()
    if p.relative_to(OUT).as_posix() not in valid_masks:
        p.unlink()  # Superseded generated preview; the previous revision retains its original.
for name in ['index.html','style.css','app.js']:
    shutil.copy2(H/name,OUT/name)
for name in ['seed_manifest.json','audit.json','semantic_exclusions.json']:
    shutil.copy2(H/name,OUT/name)
print('BUILT',OUT,'groups',len(data['groups']),'masks',len(data['seeds']))
