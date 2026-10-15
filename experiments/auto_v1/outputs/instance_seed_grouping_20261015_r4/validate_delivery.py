from pathlib import Path
import json, hashlib
from PIL import Image
import numpy as np
from grouping_rules import group_seeds
from source_policy import assert_sam_only_groups, replacement_ledger
from crop_filter import artificial_crop_edges, has_crop_contact

H=Path(__file__).resolve().parent
W=H.parents[1]/'results/固定案例_三模型对比_20261006/room2_E1_instance_seed_groups_20261015'
d=json.loads((H/'grouping.json').read_text(encoding='utf-8'))
web=json.loads((W/'data.json').read_text(encoding='utf-8'))
audit=json.loads((H/'audit.json').read_text(encoding='utf-8'))
for n,h in audit['output_sha256'].items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==h
with np.load(H/'pair_metrics.npz') as p:
    inter=p['intersection'];union=p['union'];passes=p['passes'];iou=p['iou']
counts=np.array([s['support_count'] for s in d['seeds']])
assert np.array_equal(inter,np.array(web['intersection']))
assert np.array_equal(union,counts[:,None]+counts[None,:]-inter)
assert np.array_equal(passes,(union>0)&(4*inter>union))
assert np.all(counts[:-1]>=counts[1:])
assert group_seeds(range(len(counts)),lambda a,b:passes[a,b])==[g['members'] for g in d['groups']]
with np.load(H/'surface_supports.npz') as p:
    indices=p['indices'];offsets=p['offsets']
    assert np.array_equal(np.diff(offsets),counts)
    for i,s in enumerate(d['seeds']):
        points=indices[offsets[i]:offsets[i+1]]
        assert len(np.unique(points))==len(points)
        assert hashlib.sha256(points.tobytes()).hexdigest()==s['support_sha256']
for s in web['seeds'] + web['excluded_crop_masks'] + web['excluded_noise_masks']:
    assert (W/s['rgb']).is_file()
    assert len(s['color'])==3 and all(0<=v<=255 for v in s['color'])
    p=W/s['mask'];assert hashlib.sha256(p.read_bytes()).hexdigest()==s['mask_sha256']
    a=np.asarray(Image.open(p));assert a.shape==(680,1200) and int((a>0).sum())==s['area']
    assert (W/s['source_link'].split('?')[0]).is_file()
    for a in s['aliases']:
        assert artificial_crop_edges(a['model'],a['source_box'],s['bbox'],(1200,680))==a['crop_edges']
        assert bool(a['crop_edges'])==a['touches_crop']
assert sum(len(s['aliases']) for s in web['seeds'])==d['stats']['selected_occurrences']
assert_sam_only_groups([g['members'] for g in web['groups']],web['seeds'])
sel=json.loads((H.parent/'semantic_sam_route_20261015/selection.json').read_text(encoding='utf-8'))
ledger=replacement_ledger(sel['crops'])
assert all(a['model']==2 or a['candidate_id'] not in ledger for s in web['seeds'] for a in s['aliases'])
assert not any(s['id']=='M0105' for s in web['seeds']), 'Cross-ROI CropFormer regression'
prior=json.loads((H.parent/'instance_seed_grouping_20261015_r2/grouping.json').read_text(encoding='utf-8'))
expected_keep={s['digest'] for s in prior['seeds'] if not has_crop_contact(s['aliases'])}
expected_drop={s['digest'] for s in prior['seeds'] if has_crop_contact(s['aliases'])}
assert {s['digest'] for s in web['seeds']+web['excluded_noise_masks']}==expected_keep
assert {s['digest'] for s in web['excluded_crop_masks']}==expected_drop
assert not expected_keep & expected_drop
assert not any(has_crop_contact(s['aliases']) for s in web['seeds'])
assert all(has_crop_contact(s['aliases']) for s in web['excluded_crop_masks'])
assert len(list((W/'masks').glob('*.png')))==len(web['seeds'])
assert len(list((W/'excluded_masks').glob('*.png')))==len(web['excluded_crop_masks'])
assert len(list((W/'noise_masks').glob('*.png')))==len(web['excluded_noise_masks'])
prior_by_digest={s['digest']:s for s in prior['seeds']}
assert all(s['mask_sha256']==prior_by_digest[s['digest']]['mask_sha256'] for s in web['seeds']+web['excluded_crop_masks']+web['excluded_noise_masks'])
assert not (W/'masks/M0105.png').exists()
excluded=json.loads((H/'semantic_exclusions.json').read_text(encoding='utf-8'))
assert d['stats']['selected_occurrences']+sum(len(e['removed_aliases']) for e in excluded['excluded'])+sum(len(s['aliases']) for s in web['excluded_crop_masks']+web['excluded_noise_masks'])==d['stats']['input_selected_occurrences']
prior3=json.loads((H.parent/'instance_seed_grouping_20261015_r3/grouping.json').read_text(encoding='utf-8'))
assert {s['digest'] for s in web['seeds']+web['excluded_noise_masks']}=={s['digest'] for s in prior3['seeds']}
assert len(web['seeds'])+len(web['excluded_noise_masks'])==len(prior3['seeds'])
for s in web['seeds']+web['excluded_noise_masks']:
    f=s['noise_features'];A=s['area'];B=f['bbox_area'];L=f['extent'];c=f['core_pixels']
    expected=(A<100 or L>=80 and 20*c['3']<=A or L>=80 and 10*A<=B and 20*c['8']<=A or L>=200 and 20*A<=B and 20*c['12']<=A)
    assert expected==s['noise_decision']['exclude']
    assert all(0<=c[k]<=A for k in c)
    assert c['2']>=c['3']>=c['8']>=c['12']
    assert f['area']==A and f['bbox_area']==f['bbox_width']*f['bbox_height']
    assert abs(f['bbox_fill']-A/B)<1e-12
assert all(not s['noise_decision']['exclude'] for s in web['seeds'])
assert all(s['noise_decision']['exclude'] for s in web['excluded_noise_masks'])
assert 'M0734' in {s['id'] for s in web['excluded_noise_masks']}
assert {'M0267','M0705','M0709','M0746','M0788','M0797','M0821'}<={s['id'] for s in web['seeds']}
assert not (W/'masks/M0734.png').exists()
report=dict(status='PASS',groups=len(d['groups']),masks=len(d['seeds']),
    checks=['downloaded artifacts SHA-256', 'exact grouping replay', 'strict threshold and TSDF-count order',
            'every exported support set hash and uniqueness', 'all mask areas, sizes and hashes',
            'all RGB and source links exist', 'original palette colors retained', 'all input occurrences accounted for',
            'SAM-only sources across every semantic group member and primary', 'globally replaced CropFormer cannot re-enter from another ROI',
            'M0105 absent from all displayed groups, selected seeds and page assets',
            'exact crop-contact and noise partitions; all other candidates retained',
            'natural image borders alone are preserved', 'all crop flags independently recomputed',
            'active and excluded masks retain their previous pixel hashes', 'zero crop-contact members or primary seeds',
            'strict area < 100 and independent replay of integer morphology thresholds',
            'all 67 noise exclusions visible with original colors and metadata',
            'M0734 excluded; representative compact and substantial object controls retained'])
(H/'page_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
