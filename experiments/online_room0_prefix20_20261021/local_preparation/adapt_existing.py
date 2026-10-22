"""Adapt input paths and cardinalities; preserve the accepted filtering functions."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
REF = HERE / 'reference'
OUT = HERE / 'code'
C = '/home/chenkejun/CVPR/experiments/online_room0_prefix20_20261021'
R = '/data/chenkejun/CVPR/results/online_room0_prefix20_20261021'
STAGES = ['frontend','overmerge','intersection','versions','semantic_route','grouping_r2','grouping_r3','v1','v2']
REPLACE = {
 '/data/chenkejun/CVPR/results/auto_v2_room0_20261019': R,
 '/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019': C,
 '/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room0/abstain': R + '/baseline',
 '/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/catalog': R + '/catalog',
 '/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010': C + '/catalog',
 '/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/configs/raw.json': C + '/config.json',
 '/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/room0/geometry/geometry_full400.json': R + '/geometry/geometry_prefix20.json',
 '/data/chenkejun/CVPR/datasets/cropformer_frozen/room0': R + '/inputs/masks',
 '/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/room0': R + '/unused_historical_cache',
}
changes = []

def write(rel, original, text):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == '.py': ast.parse(text, filename=str(p))
    elif p.suffix == '.json': json.loads(text)
    p.write_text(text, encoding='utf-8')
    changes.append(dict(file=str(rel).replace('\\','/'), original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(text.encode()).hexdigest(),
        diff=''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='reference/'+str(rel), tofile='online/'+str(rel)))))

def replace_once(text, old, new):
    assert old in text, old[:180]
    return text.replace(old, new)

for stage in STAGES:
    for source in (REF / 'upstream' / stage).rglob('*'):
        if source.suffix not in ['.py', '.json']: continue
        rel = source.relative_to(REF / 'upstream')
        original = source.read_text(encoding='utf-8')
        text = original
        for a,b in sorted(REPLACE.items(), key=lambda x:-len(x[0])): text=text.replace(a,b)
        if rel.as_posix() == 'frontend/prepare_inputs.py':
            text = text.replace("NAME='room0-e1-prerepair-CUT-top20-20261015'", "NAME='room0-prefix20-online-ROI-20261021'")
            text = text.replace(",OLD/'input_manifest.json'", '')
            text = text.replace("entries=[e for e in catalog['entries'] if e['rank']<=20]", "entries=[e for e in catalog['entries'] if e['rank']<=20 and e['views']]\n nentries=len(entries); ncases=len({e['uid'] for e in entries}); nviews=sum(len(e['views']) for e in entries)\n assert nentries>0, 'No viewable prefix ROI'")
            text = text.replace("assert len(entries)==60 and all(sum(e['group']==g for e in entries)==20 for g in ['CONFLICT','U','T'])", "assert len(entries)<=60 and all(sum(e['group']==g for e in entries)<=20 for g in ['CONFLICT','U','T'])")
            text = text.replace("assert len({e['uid'] for e in entries})==34 and all(len(e['views'])==3 for e in entries)", "assert all(1<=len(e['views'])<=3 and all(v['frame'] in range(0,96,5) for v in e['views']) for e in entries)")
            text = text.replace("old=read(OLD/'input_manifest.json');old_cases={}", "old={'jobs':[],'policy':{'candidate_inputs':'prefix-only ROI; unchanged balanced/context/tile crop rules','old_sequence_cache_used':False}};old_cases={}")
            text = text.replace('unique_selected_ROIs=34','unique_selected_ROIs=ncases').replace('selected_ROIs=34','selected_ROIs=ncases').replace('ROI_count=34','ROI_count=ncases').replace('selected_cases=34','selected_cases=ncases').replace('cases=34','cases=ncases')
            text = text.replace('ranked_entries=60','ranked_entries=nentries').replace('preferred_views=180','preferred_views=nviews').replace('views=180','views=nviews')
            text = text.replace("len(cases)==34 and sum(len(c['views']) for c in cases)==180", "len(cases)==ncases and sum(len(c['views']) for c in cases)==nviews")
            text = text.replace("sum(len(c['views']) for c in cases)==180", "sum(len(c['views']) for c in cases)==nviews")
            text = text.replace('60 entries / 34 unique components','up to 20 per state; actual prefix counts')
        elif rel.as_posix() == 'frontend/normalize_manifest.py':
            text = text.replace("if len(manifest['items'])==34:", "if len(manifest['items'])==len({x['uid'] for x in manifest['items']}):")
            text = text.replace("assert len(manifest['items'])==60", "assert len(manifest['items'])==inputs['ranked_entries']")
            text = text.replace('assert len(items)==34', "assert len(items)==inputs['selected_cases']")
        elif rel.as_posix() == 'frontend/summarize_frontend.py':
            text = text.replace("scene_counts={'room0':34},total_views=180", "scene_counts={'room0':m['selected_cases']},total_views=m['preferred_views']")
            text = text.replace("len(viewer['cases'])==34 and sum(len(c['views']) for c in viewer['cases'])==180", "len(viewer['cases'])==m['selected_cases'] and sum(len(c['views']) for c in viewer['cases'])==m['preferred_views']")
            text = text.replace('cases=34,views=180', "cases=m['selected_cases'],views=m['preferred_views']")
        elif rel.as_posix() == 'intersection/compute.py':
            text = text.replace("len(out['cases'])==len(d['cases'])==34", "len(out['cases'])==len(d['cases'])")
        elif rel.as_posix() == 'grouping_r2/source_overrides.json':
            text = json.dumps(dict(scene='room0',kind='prefix_only_no_historical_mask_overrides', corrections=[],
                reason='Full-sequence M0466 identity cannot be applied to fresh prefix candidate IDs.'),indent=2)
        elif rel.as_posix() == 'v2/filter_rules.py':
            text = (REF / 'swap_rules/filter_rules.py').read_text(encoding='utf-8')
        elif rel.as_posix() == 'v2/measure_and_select.py':
            text = text.replace('from coverage import measure', 'from coverage import measure\nfrom semantic_exemption import apply_semantic_exemption')
            text = text.replace("CASE='room0_auto_v2_20261019'", "CASE='room0_online_prefix20_20261021'")
            text = text.replace('try:selection=run_filters(data)', 'try:selection=apply_semantic_exemption(run_filters(data),sem)')
        write(rel, original, text)

source = REF / 'swap_rules/semantic_exemption.py'
write('v2/semantic_exemption.py', source.read_text(), source.read_text())
original = (REF / 'catalog/common.py').read_text(encoding='utf-8')
text = original
start = text.index('def paths(scene):')
end = text.index('def project_visible',start)
text = text[:start] + f'''def paths(scene):
 assert scene=='room0'
 return (Path('{R}/baseline/final/instance_surface.npz'),Path('{R}/baseline/surface_evidence.npz'),
         Path('{R}/geometry/review_surface_graph_n02_p08.npz'),Path('{C}/config.json'))

''' + text[end:]
text = text.replace('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/catalog', R+'/catalog')
write('catalog/common.py',original,text)
original=(REF/'catalog/build_candidates.py').read_text(encoding='utf-8')
text=original
text=re.sub(r" freeze=read\('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/predictions_freeze.json'\).*\n assert str\(surface\)==freeze\['path'\] and sha\(surface\)==freeze\['sha256'\]",
    f" assert read('{R}/mapping_complete.json')['frame_ids']==list(range(0,96,5))",text)
assert 'predictions_freeze.json' not in text
text=text.replace('[:30] for k in GROUPS','[:20] for k in GROUPS')
text=text.replace('for fid in range(len(poses)):', 'for fid in range(0,96,5):')
text=text.replace("e['all_frames_checked']=len(poses)", "e['all_frames_checked']=20")
text=text.replace("history_frames':len(poses)", "history_frames':20")
text=text.replace("'top30_each_exact':True", "'up_to_top20_each_exact':True")
text=text.replace("'U':'#757d89'", "'U':'#000000'")
text=text.replace('完整历史序列中未找到','前20个处理帧中未找到')
write('catalog/build_candidates.py',original,text)
(HERE/'adapter_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Adapted',len(changes),'files; filtering functions preserved')
