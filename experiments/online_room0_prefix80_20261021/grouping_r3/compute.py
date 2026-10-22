"""Read-only grouping of selected room0 masks on the frozen TSDF surface."""
from pathlib import Path
from collections import OrderedDict
import sys, json, hashlib, zipfile, random, time
import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from grouping_rules import same_instance, group_seeds
from source_policy import replacement_ledger, eligible_aliases, assert_sam_only_groups
from crop_filter import artificial_crop_edges, has_crop_contact

C = Path(__file__).resolve().parent
R = Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/grouping_r3')
F = Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/overmerge')
BASE = Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021/baseline/final/instance_surface.npz')
CAMERA = Path('/home/chenkejun/CVPR/experiments/online_room0_prefix80_20261021/config.json')
COMMON = Path('/home/chenkejun/CVPR/experiments/online_room0_prefix80_20261021/catalog/common.py')
R.mkdir(exist_ok=True, parents=True)
started = time.time()

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def save(name, data):
    (R / name).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

rule = json.loads((C / 'config.json').read_text())
assert rule['iou'] == .25 and rule['comparison'] == 'strictly greater than'
previous_path = Path(rule['previous_grouping'])
previous = json.loads(previous_path.read_text())
previous_by_id = {s['id']: s for s in previous['seeds']}
selection_path = Path(rule['input_selection'])
selected = json.loads(selection_path.read_text())
replacements = replacement_ledger(selected['crops'])
fpath = F / 'web/data.json'
f = json.loads(fpath.read_text())
expected = json.loads((F / 'input_hashes.json').read_text())
cfg = json.loads(CAMERA.read_text())
scene = Path(cfg['source']['scene_root'])
trajectory = scene / cfg['source']['trajectory']
protected = [selection_path, fpath, F / 'input_hashes.json', BASE, CAMERA, COMMON, trajectory, previous_path]
source_sha = {str(p): sha(p) for p in protected}
assert source_sha[str(BASE)] == f['source_hashes'][str(BASE)]
assert source_sha[str(fpath)] == selected['source_sha256'][str(fpath)]
occurrences = [(key, c, n, c['models'][n['model']]['labels'][str(n['label'])])
               for key, c in selected['crops'].items() for n in c['nodes']
               if c['models'][n['model']]['labels'][str(n['label'])]['status'] == 'selected']
frames = sorted({c['frame'] for _, c, _, _ in occurrences})
print('INPUT', len(occurrences), 'selected occurrences;', len(frames), 'frames', flush=True)
sys.path.insert(0, str(COMMON.parent))
import common
camera = cfg['camera']
W, H = camera['width'], camera['height']
poses = np.loadtxt(trajectory).reshape(-1, 4, 4)
with np.load(BASE) as z: xyz = z['xyz_m']
num_vertices = len(xyz)
visible = {}
for fi, frame in enumerate(frames):
    p = F / f'frame{frame:06d}_visible.npz'
    with np.load(p) as z:
        idx = z['vertex_index'].astype(np.int32)
        uv = z['uv'].astype(np.int32)
    assert len(idx) == len(np.unique(idx)) and np.all(idx[:-1] < idx[1:])
    depth_path = scene / cfg['source']['depth_pattern'].format(frame=frame)
    depth = np.asarray(Image.open(depth_path), dtype=np.float64) / camera['depth_png_units_per_meter']
    valid, fulluv = common.project_visible(xyz, poses[frame], depth, camera)
    assert np.array_equal(np.flatnonzero(valid), idx), ('visibility mismatch', frame)
    assert np.array_equal(fulluv[idx], uv), ('projection mismatch', frame)
    visible[frame] = (idx, uv)
    source_sha[str(p)] = sha(p)
    source_sha[str(depth_path)] = sha(depth_path)
    if fi % 20 == 0: print('VISIBILITY', fi + 1, '/', len(frames), flush=True)
del xyz, fulluv, valid, depth

arrays = OrderedDict()
mask_cache, unique, candidate_digests = {}, {}, {}
for oi, (key, c, n, row) in enumerate(occurrences):
    frame, model, label = int(c['frame']), int(n['model']), int(n['label'])
    source = f['sources'][n['source_index']]
    path = str(source['path'])
    assert path == n['source_path']
    cachekey = (n['source_index'], label)
    if cachekey not in mask_cache:
        if path not in arrays:
            source_sha[path] = sha(path)
            assert source_sha[path] == expected[path], ('source modified', path)
            arrays[path] = np.asarray(Image.open(path))
            if len(arrays) > 12: arrays.popitem(last=False)
        arrays.move_to_end(path)
        arr = arrays[path]
        box = list(map(int, source['box']))
        x0, y0, x1, y1 = box
        assert arr.shape == (y1-y0, x1-x0), (path, arr.shape, box)
        binary = np.zeros((H, W), dtype=bool)
        binary[y0:y1, x0:x1] = arr == label
        assert int(binary.sum()) == n['area'], ('area mismatch', key, label)
        yy, xx = np.nonzero(binary)
        assert len(xx) > 0
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
        packed = np.packbits(binary.reshape(-1)).tobytes()
        digest = hashlib.sha256(frame.to_bytes(4, 'little') + packed).hexdigest()
        idx, uv = visible[frame]
        support = idx[binary[uv[:, 1], uv[:, 0]]]
        crop_touch = bool(artificial_crop_edges(model, box, bbox, (W, H)))
        frame_touch = bbox[0] == 0 or bbox[1] == 0 or bbox[2] == W or bbox[3] == H
        mask_cache[cachekey] = (digest, bbox, len(xx), packed, support, bool(crop_touch), bool(frame_touch))
    digest, bbox, area, packed, support, crop_touch, frame_touch = mask_cache[cachekey]
    assert len(support) == int(n['support']['N']), ('support mismatch', key, n['key'], len(support), n['support']['N'])
    cid = n['candidate_id']
    assert cid not in candidate_digests or candidate_digests[cid] == digest
    candidate_digests[cid] = digest
    alias = dict(crop_key=key, ROI=c['ROI'], uid=c['uid'], frame=frame, model=model, label=label,
                 candidate_id=cid, source_index=n['source_index'], source_path=path, source_box=source['box'],
                 selection_reason=row['reason'], semantic_hit=bool(n.get('semantic_hit')),
                 semantic_family=(n.get('semantic') or {}).get('target_family'),
                 touches_crop=crop_touch, touches_frame=frame_touch,
                 crop_edges=artificial_crop_edges(model, source['box'], bbox, (W, H)))
    if digest not in unique:
        unique[digest] = dict(frame=frame, area=area, bbox=bbox, digest=digest, packed=packed,
                              support=support, aliases=[])
    else:
        assert unique[digest]['packed'] == packed and np.array_equal(unique[digest]['support'], support)
    unique[digest]['aliases'].append(alias)
    if oi % 250 == 0: print('MASKS', oi + 1, '/', len(occurrences), flush=True)
del arrays, mask_cache, visible

for s in unique.values():
    s['representative'] = min(s['aliases'], key=lambda a: (not a['semantic_hit'], a['model'], a['touches_crop'], a['crop_key'], a['label']))
    s['support_count'] = len(s['support'])
seeds = sorted(unique.values(), key=lambda s: (-s['support_count'], s['frame'], s['representative']['model'], s['digest']))
for i, s in enumerate(seeds): s.update(index=i, id=f'M{i+1:04d}')
input_unique_masks = len(seeds)
excluded = []
eligible = []
for s in seeds:
    aliases = eligible_aliases(s['aliases'], replacements)
    removed = [a for a in s['aliases'] if a not in aliases]
    if removed:
        excluded.append(dict(id=s['id'], digest=s['digest'], frame=s['frame'], support_count=s['support_count'],
            area=s['area'], removed_aliases=removed, entire_mask_excluded=not aliases,
            reason='GLOBAL_SEMANTIC_SAM_ONLY',
            replacement_evidence={a['candidate_id']: replacements[a['candidate_id']]
                                  for a in removed if a['candidate_id'] in replacements}))
    if aliases:
        s['aliases'] = aliases
        s['representative'] = min(aliases, key=lambda a: (not a['semantic_hit'], a['model'], a['touches_crop'], a['crop_key'], a['label']))
        eligible.append(s)
# Carry the verified previous source decisions through the independent pixel
# reconstruction. This preserves the explicit M0466 correction and prevents a
# cross-view SAM-only rejection from returning at the crop-filter stage.
source_report_path=previous_path.parent/'semantic_exclusions.json'
source_sha[str(source_report_path)]=sha(source_report_path)
source_report=json.loads(source_report_path.read_text())
rejected={e['id']:e for e in source_report['excluded'] if e['entire_mask_excluded']}
verified_eligible=[]
for s in eligible:
    if s['id'] not in previous_by_id:
        assert s['id'] in rejected and rejected[s['id']]['digest']==s['digest']
        if not any(e['id']==s['id'] for e in excluded):excluded.append(rejected[s['id']])
        continue
    prior=previous_by_id[s['id']]
    assert prior['digest']==s['digest'] and prior['support_sha256']==hashlib.sha256(s['support'].tobytes()).hexdigest()
    assert {(a['candidate_id'],a['crop_key'],a['label']) for a in prior['aliases']} <= {(a['candidate_id'],a['crop_key'],a['label']) for a in s['aliases']}
    s['aliases']=prior['aliases'];s['representative']=prior['representative']
    if 'source_policy_user_override' in prior:s['source_policy_user_override']=prior['source_policy_user_override']
    verified_eligible.append(s)
eligible=verified_eligible
source_eligible_masks = len(eligible)
crop_excluded = [s for s in eligible if has_crop_contact(s['aliases'])]
seeds = [s for s in eligible if not has_crop_contact(s['aliases'])]
assert {s['digest'] for s in eligible} == {s['digest'] for s in previous['seeds']}
assert not any(has_crop_contact(s['aliases']) for s in seeds)
for i, s in enumerate(seeds): s['index'] = i
assert all(a['model'] == 2 or a['candidate_id'] not in replacements for s in seeds for a in s['aliases'])
print('SAM_SOURCE_POLICY', len(excluded), 'source exclusions;', len(seeds), 'eligible masks', flush=True)
print('CROP_EDGE_FILTER', len(crop_excluded), 'masks removed;', len(seeds), 'remaining', flush=True)
counts = np.array([s['support_count'] for s in seeds], dtype=np.int64)
offsets = np.concatenate(([0], np.cumsum(counts)))
indices = np.concatenate([s['support'] for s in seeds]).astype(np.int32)
matrix = csr_matrix((np.ones(len(indices), np.int32), indices, offsets), shape=(len(seeds), num_vertices))
inter = (matrix @ matrix.T).toarray().astype(np.int64)
union = counts[:, None] + counts[None, :] - inter
iou = np.divide(inter, union, out=np.zeros(inter.shape, float), where=union > 0)
passes = (union > 0) & (4 * inter > union)
assert np.array_equal(inter, inter.T) and np.array_equal(np.diag(inter), counts)
groups_raw = group_seeds(range(len(seeds)), lambda a, b: bool(passes[a, b]))
assert_sam_only_groups(groups_raw, seeds)
group_of = {s: gi for gi, members in enumerate(groups_raw) for s in members}
groups = []
for gi, members in enumerate(groups_raw):
    primary = members[0]
    assert counts[primary] == max(counts[members])
    assert all(passes[primary, j] for j in members[1:])
    for j in members[1:]:
        exact = len(np.intersect1d(seeds[primary]['support'], seeds[j]['support'], assume_unique=True))
        assert exact == inter[primary, j] and same_instance(exact, counts[primary], counts[j])
    vframes, sameframe = {}, []
    for j in members[1:]:
        if seeds[j]['frame'] == seeds[primary]['frame']: sameframe.append(j)
        else: vframes.setdefault(seeds[j]['frame'], []).append(j)
    aux_below = sum(not passes[a, b] for ai, a in enumerate(members[1:]) for b in members[ai+2:])
    warnings = []
    assert not any(a['touches_crop'] for j in members for a in seeds[j]['aliases'])
    if counts[primary] == 0: warnings.append('zero_tsdf_support')
    if aux_below: warnings.append('auxiliary_pairs_below_threshold_allowed')
    nearest = sorted((j for j in range(len(seeds)) if group_of[j] != gi and inter[primary, j] > 0),
                     key=lambda j: (-iou[primary, j], j))[:5]
    all_pair_values = [float(iou[a, b]) for ai, a in enumerate(members) for b in members[ai+1:]]
    groups.append(dict(id=f'G{gi+1:03d}', index=gi, primary=primary, members=members, member_count=len(members),
        frames=sorted({seeds[j]['frame'] for j in members}),
        validation_frames=[dict(frame=fr, members=js, representative=js[0]) for fr, js in sorted(vframes.items())],
        same_frame_references=sameframe, min_primary_iou=min((float(iou[primary, j]) for j in members[1:]), default=None),
        min_all_pair_iou=min(all_pair_values, default=None), below_threshold_aux_pairs=int(aux_below), warnings=warnings,
        nearest_outside=[dict(member=j, group=group_of[j], iou=float(iou[primary, j])) for j in nearest],
        semantic_window_present=any(a['semantic_hit'] for j in members for a in seeds[j]['aliases']),
        semantic_families=sorted({a['semantic_family'] for j in members for a in seeds[j]['aliases'] if a['semantic_hit']})))

(R / 'masks').mkdir(exist_ok=True)
(R / 'excluded_masks').mkdir(exist_ok=True)
flat = []
for s in seeds:
    binary = np.unpackbits(np.frombuffer(s['packed'], np.uint8), count=H*W).reshape(H, W)
    assert int(binary.sum()) == s['area']
    rel = f"masks/{s['id']}.png"
    Image.fromarray(binary * 255).save(R / rel)
    node = {k: v for k, v in s.items() if k not in ['packed', 'support']}
    node.update(group=group_of[s['index']], mask=rel, mask_sha256=sha(R/rel),
                support_sha256=hashlib.sha256(s['support'].tobytes()).hexdigest(),
                rois=sorted({a['uid'] for a in s['aliases']}), candidate_ids=sorted({a['candidate_id'] for a in s['aliases']}))
    flat.append(node)
flat_excluded = []
for i, s in enumerate(crop_excluded):
    binary = np.unpackbits(np.frombuffer(s['packed'], np.uint8), count=H*W).reshape(H, W)
    rel = f"excluded_masks/{s['id']}.png"
    Image.fromarray(binary * 255).save(R/rel)
    old = previous_by_id[s['id']]
    old_group = previous['groups'][old['group']]
    node = {k: v for k, v in s.items() if k not in ['packed', 'support', 'index']}
    node.update(index=i, group=None, mask=rel, mask_sha256=sha(R/rel),
        reason='ARTIFICIAL_CROP_EDGE_CONTACT', previous_group=old_group['id'],
        previous_primary=old_group['primary']==old['index'],
        rois=sorted({a['uid'] for a in s['aliases']}),
        candidate_ids=sorted({a['candidate_id'] for a in s['aliases']}))
    assert has_crop_contact(node['aliases'])
    flat_excluded.append(node)
active_ids = {s['id'] for s in seeds}
emptied_groups = [dict(id=g['id'], mask_ids=[previous['seeds'][i]['id'] for i in g['members']])
    for g in previous['groups'] if not any(previous['seeds'][i]['id'] in active_ids for i in g['members'])]
np.savez_compressed(R / 'surface_supports.npz', indices=indices, offsets=offsets)
np.savez_compressed(R / 'pair_metrics.npz', intersection=inter, union=union, iou=iou, passes=passes)
stats = dict(input_selected_occurrences=len(occurrences), input_unique_masks=input_unique_masks,
    selected_occurrences=sum(len(s['aliases']) for s in seeds), source_candidates=len(candidate_digests), unique_masks=len(seeds),
    excluded_cf_masks=sum(e['entire_mask_excluded'] for e in excluded),
    excluded_cf_occurrences=sum(len(e['removed_aliases']) for e in excluded),
    before_crop_filter_masks=source_eligible_masks,
    excluded_crop_masks=len(crop_excluded), excluded_crop_occurrences=sum(len(s['aliases']) for s in crop_excluded),
    previous_groups_emptied=len(emptied_groups), previous_primaries_removed=sum(s['previous_primary'] for s in flat_excluded),
    exact_duplicate_occurrences_removed=sum(len(s['aliases']) for s in seeds)-len(seeds), frames=len(frames), rois=len({a['uid'] for s in seeds for a in s['aliases']}),
    groups=len(groups), multi_groups=sum(len(g['members'])>1 for g in groups), singleton_groups=sum(len(g['members'])==1 for g in groups),
    zero_support_masks=int(sum(counts==0)), primary_seeds=len(groups), auxiliary_masks=len(seeds)-len(groups),
    groups_without_other_frame=sum(not g['validation_frames'] for g in groups),
    groups_with_low_aux_pair=sum(g['below_threshold_aux_pairs']>0 for g in groups),
    semantic_groups=sum(g['semantic_window_present'] for g in groups),
    semantic_groups_non_sam_primary=sum(g['semantic_window_present'] and seeds[g['primary']]['representative']['model']!=2 for g in groups),
    semantic_groups_cf_members=sum(seeds[i]['representative']['model']!=2 for g in groups if g['semantic_window_present'] for i in g['members']))
assert stats['semantic_groups_non_sam_primary'] == 0 and stats['semantic_groups_cf_members'] == 0
visibility_rule = dict(function=str(COMMON)+':project_visible', depth_tolerance=float(common.DEPTH_TOLERANCE),
                       verified_frames=len(frames), vertex_count=num_vertices, image_width=W, image_height=H)
save('grouping.json', dict(rule=rule, stats=stats, seeds=flat, groups=groups, intersection=inter.tolist(),
                          excluded_crop_masks=flat_excluded, source_sha256=source_sha, visibility_rule=visibility_rule))
save('crop_exclusions.json', dict(policy=rule['crop_contact_policy'], stats=stats, excluded=flat_excluded,
    previous_groups_emptied=emptied_groups))
save('semantic_exclusions.json', dict(policy=rule['source_policy'], excluded=excluded,
    previously_replaced_candidate_count=len(replacements),manual_source_corrections=source_report.get('manual_source_corrections'),source_resolution_passes=source_report.get('source_resolution_passes'),stats=stats))
save('seed_manifest.json', dict(status='awaiting_visual_review', tracking_requested=False, map_write=False, rule=rule,
    groups=[dict(id=g['id'], primary=flat[g['primary']], auxiliary_ids=[flat[j]['id'] for j in g['members'][1:]],
                 validation_frames=g['validation_frames']) for g in groups]))
rng = random.Random(20261015)
for _ in range(150):
    a, b = rng.randrange(len(seeds)), rng.randrange(len(seeds))
    exact = len(set(map(int, seeds[a]['support'])) & set(map(int, seeds[b]['support'])))
    assert exact == inter[a,b]
    assert bool(passes[a,b]) == same_instance(exact, int(counts[a]), int(counts[b]))
assert sorted(j for g in groups for j in g['members']) == list(range(len(seeds)))
assert len(flat) == len({s['digest'] for s in flat})
for p, digest in source_sha.items(): assert sha(p) == digest, ('input changed during run', p)
audit = dict(status='PASS', stats=stats, elapsed_seconds=round(time.time()-started, 2),
    checks=['all source file hashes unchanged', 'all frame visibility caches independently recomputed',
            'every support count matches prior projection', 'exact-mask duplicate aliases retained',
            'unique TSDF indices only; no old instance IDs used', 'strict IoU > 0.25 via integer comparison',
            'fixed-primary grouping; no transitive or all-pairs gate', 'all accepted primary edges independently recounted',
            '150 random pairs independently recounted', 'all eligible masks assigned exactly once; max-TSDF primary each group',
            'semantic CropFormer replacement propagated across all ROI aliases',
            'zero CropFormer members and primaries in all window/cabinet groups',
            'zero artificial-crop-contact members or primaries',
            'active plus crop-excluded masks exactly reproduce previous source-eligible input'],
    output_sha256={n: sha(R/n) for n in ['grouping.json','seed_manifest.json','surface_supports.npz','pair_metrics.npz','semantic_exclusions.json','crop_exclusions.json']})
save('audit.json', audit)
with zipfile.ZipFile(R/'grouping_data.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2) as z:
    for name in ['grouping.json','audit.json','seed_manifest.json','surface_supports.npz','pair_metrics.npz','semantic_exclusions.json','crop_exclusions.json']:
        z.write(R/name, name)
    for p in sorted((R/'masks').glob('*.png')): z.write(p, p.relative_to(R).as_posix())
    for p in sorted((R/'excluded_masks').glob('*.png')): z.write(p, p.relative_to(R).as_posix())
save('receipt.json', dict(sha256=sha(R/'grouping_data.zip')))
print('COMPLETE', json.dumps(stats), flush=True)
