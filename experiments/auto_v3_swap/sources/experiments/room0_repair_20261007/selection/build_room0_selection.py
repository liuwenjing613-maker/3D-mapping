"""Prepare an isolated room0 human-selection page from frozen model outputs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
SOURCE = WORKSPACE / 'results' / '固定案例_三模型对比_20261006'
DEST = SOURCE / 'room0_selection_20261007'
TEMPLATE = WORKSPACE / 'outputs' / 'room2_selection_20261007' / 'room2_selection_template.html'
OLD_SELECTION = WORKSPACE / '修复候选选择_20261006.json'


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def replace_once(text: str, old: str, new: str):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


def main():
    protected = [SOURCE / 'select.html', SOURCE / 'viewer_data.json',
                 SOURCE / 'projection_annotations.json', OLD_SELECTION, TEMPLATE,
                 SOURCE / 'room2_selection_20261007' / 'index.html',
                 WORKSPACE / 'room2_修复候选选择_20261007.json']
    protected_hashes = {str(p.relative_to(WORKSPACE)): sha(p) for p in protected}
    data = read_json(SOURCE / 'viewer_data.json')
    cases = [c for c in data['cases'] if c['scene'] == 'room0']
    assert len(cases) == 14
    uids = [c['uid'] for c in cases]
    assert len(set(uids)) == len(uids)
    view_count = sum(len(c['views']) for c in cases)
    assert view_count == 42
    keys = {k for c in cases for v in c['views'] for k in v['variants']}
    for c in cases:
        for v in c['views']:
            assert v['recommended'] in v['variants']
    scoped_data = dict(data, name='room0 人工修复选择', cases=cases,
                       crops={k: v for k, v in data['crops'].items() if k in keys},
                       scene_counts={'room0': len(cases)}, total_views=view_count)
    projections = read_json(SOURCE / 'projection_annotations.json')
    scoped_projections = dict(projections,
        crops={k: v for k, v in projections['crops'].items() if k in keys},
        views={k: v for k, v in projections['views'].items()
               if any(k.startswith(uid + '_') for uid in uids)})
    assert set(scoped_data['crops']) == keys == set(scoped_projections['crops'])
    asset_paths = set()

    def prefix_assets(value):
        if isinstance(value, dict):
            return {k: prefix_assets(v) for k, v in value.items()}
        if isinstance(value, list):
            return [prefix_assets(v) for v in value]
        if isinstance(value, str) and value.startswith(('assets/', 'comparisons/')):
            asset_paths.add(value)
            return '../' + value
        return value

    scoped_data = prefix_assets(scoped_data)
    scoped_projections = prefix_assets(scoped_projections)
    missing = [p for p in sorted(asset_paths) if not (SOURCE / p).is_file()]
    assert not missing, missing[:10]
    for crop in scoped_data['crops'].values():
        assert len(crop['models']) == 3
        for model in crop['models']:
            assert model['ids'] and model['ids_file']
            # The frozen renderer places text labels only on masks >=20 pixels.
            assert model['count'] == len(model['colors']) >= len(model['positions'])
            assert len({p['id'] for p in model['positions']}) == len(model['positions'])
            assert all(p['id'] > 0 and str(p['id']) in model['colors'] for p in model['positions'])

    old = read_json(OLD_SELECTION)
    references = {s['case_uid']: s for s in old['selections'] if s['scene'] == 'room0'}
    assert set(references).issubset(set(uids))
    template = TEMPLATE.read_text(encoding='utf-8').replace('room2', 'room0')
    template = replace_once(template, '5 个固定问题区域 · 15 个视角',
                            f'{len(cases)} 个固定问题区域 · {view_count} 个视角')
    template = replace_once(template, 'room0 · 全部 5 个区域',
                            f'room0 · 全部 {len(cases)} 个区域')
    template = replace_once(template, '5 个区域全部完成。', f'{len(cases)} 个区域全部完成。')
    a = template.index('const previousReferences=')
    b = template.index('\nfunction refreshPreviousReference', a)
    template = template[:a] + 'const previousReferences=' + json.dumps(references, ensure_ascii=False) + ';' + template[b:]
    template, matches = re.subn(r'const pilotUIDs=new Set\(\[.*?\]\);',
                               'const pilotUIDs=new Set(' + json.dumps(uids) + ');',
                               template, count=1)
    assert matches == 1
    workflow = ('你选完整 mask → 我检查跨区域是否是同一物体 → 全序列双向追踪 → '
                '合并修复 room0 → 同视角 3D 前后对比和统一 v3 评估')
    template = replace_once(template, workflow,
                            '当前：等待你标注，尚未开始追踪。每区选一个最佳视角和模型，可选多个完整物体；'
                            '保存全部区域后导出 JSON → 核对重复物体 → 双向 SAM2.1 追踪 → 联合换票 → 3D 与统一 v3 对比')
    assert 'room2' not in template
    assert '5 个区域' not in template and '15 个视角' not in template
    assert "'cvpr-room0-selection-f49eb927-20261007-v1'" in template
    assert "scene_scope:'room0'" in template
    assert 'room0_修复候选选择_20261007.json' in template
    page = template.replace('__DATA__', json.dumps(scoped_data, ensure_ascii=False).replace('</', '<\\/'))
    page = page.replace('__PROJECTIONS__', json.dumps(scoped_projections, ensure_ascii=False).replace('</', '<\\/'))
    assert '__DATA__' not in page and '__PROJECTIONS__' not in page
    runtime = re.findall(r'<script>\s*(.*?)</script>', page, re.S)
    assert len(runtime) == 1
    CODE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    (CODE / 'room0_selection_template.html').write_text(template, encoding='utf-8')
    (CODE / 'room0_selection_runtime.js').write_text(runtime[0], encoding='utf-8')
    (DEST / 'index.html').write_text(page, encoding='utf-8')
    write_json(DEST / 'viewer_data.json', scoped_data)
    write_json(DEST / 'previous_room0_reference.json', references)
    setup = {
        'status': 'READY_FOR_HUMAN_CHOICES', 'scene_scope': 'room0',
        'source_selection': 'selection-20261006-f49eb927', 'case_count': len(cases),
        'view_count': view_count, 'candidate_crop_count': len(keys),
        'models': ['original', 'cropformer_local', 'sam2_1_local'],
        'cases': [{'uid': c['uid'], 'ROI': c['ROI'], 'frames': [v['frame'] for v in c['views']]} for c in cases],
        'existing_assets_reused': len(asset_paths), 'missing_assets': missing,
        'isolated_storage_key': 'cvpr-room0-selection-f49eb927-20261007-v1',
        'previous_choices_are_reference_only': True, 'previous_reference_count': len(references),
        'human_records_created': 0, 'model_inference_rerun': False,
        'tracking_started': False, 'map_modified': False,
        'export_filename': 'room0_修复候选选择_20261007.json',
        'original_files_sha256': protected_hashes,
        'crop_border_policy': 'allow save; retain touching mask IDs for seed validation',
        'next': ['wait for new human selections', 'validate seed masks and cross-ROI identities without GT',
                 'SAM2.1 independent forward and reverse tracking over 2000 raw frames',
                 'joint replacement on original 400 mapping-frame votes',
                 'verify full replay, exact rollback and zero outside-region changes',
                 'compare baseline, seed-only and full-track repair with unchanged unified v3',
                 'show stable-camera colored PLY with individually visible instance IDs'],
        'evaluation_protocol': 'unified v3; same parameters for all conditions; pending diagnostic protocol',
    }
    write_json(DEST / 'selection_setup.json', setup)
    (DEST / 'plan.txt').write_text(
        'room0 本轮人工标注\n\n'
        '14个固定问题区域，42个视角，原始CropFormer、局部CropFormer、SAM2.1三种候选并排。\n'
        '红色提示标出冲突点对应的位置；可切换未知点、暂定点或合并查看。\n'
        '每个区域选一个最佳视角、裁剪和模型，再点选要修复的完整物体。\n'
        '如果两个物体被合并，分别选择对应的两个完整mask；不相关的墙、地面不要选入。\n'
        '每个mask可单独显示/隐藏、高亮、只看；三个方案都差时选“都不可靠”。\n'
        '允许保存触及人工裁剪边界的mask，导出保留触边编号，后续单独核对完整度。\n'
        '旧3处选择仅作参考，不自动计为本轮标注。不同区域若是同一物体，可在目标说明里写明。\n'
        '每区点“保存并下一例”；全部完成后点“导出room0选择”。\n'
        '将room0_修复候选选择_20261007.json放到D:\\Users\\刘雯静\\Downloads\\CVPR，再告诉Codex“room0已标好”。\n\n'
        '现在仅准备标注页，没有运行新追踪或修改地图。收到你的记录后才开始：\n'
        '核对种子和重复物体；独立向前、向后追踪2000帧；在原有400建图帧联合撤回和补票；\n'
        '验证完整重算、回滚和范围外不变；展示原图、仅种子、完整追踪三组3D与统一v3指标。\n',
        encoding='utf-8')
    for rel, digest in protected_hashes.items():
        assert sha(WORKSPACE / rel) == digest, rel
    receipt = dict(setup, output_directory=str(DEST), page_sha256=sha(DEST / 'index.html'),
                   original_files_unchanged=True)
    write_json(CODE / 'build_receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status', 'case_count', 'view_count', 'candidate_crop_count',
                                            'existing_assets_reused', 'original_files_unchanged', 'tracking_started',
                                            'output_directory']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
