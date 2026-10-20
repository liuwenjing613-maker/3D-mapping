"""Derive a read-only stage-5 counterfactual from the frozen review evidence."""
import hashlib
import json
from pathlib import Path


def build_preview(data, source_manifest, output):
    rows = {r['group_id']: r for r in data['rows']}
    groups = {g['id']: g for g in source_manifest['groups']}
    selection = data['selection']
    old = set(selection['retained_group_ids'])
    incoming = set(next(s for s in selection['stages'] if s['stage'] == 'full_support_containment')['retained_group_ids'])
    removed = set(next(s for s in selection['stages'] if s['stage'] == 'pr_filter')['removed_group_ids'])
    assert incoming == old | removed and not old & removed
    families = {'window', 'cabinet'}
    confirmed = {gid for gid in incoming
        if groups[gid]['primary']['representative'].get('semantic_hit') is True
        and groups[gid]['primary']['representative'].get('semantic_family') in families}
    broad = {gid for gid in incoming if rows[gid]['semantic']['top_class'] in {'windowpane', 'cabinet'}}
    added = sorted(confirmed & removed)
    extra = sorted((broad & removed) - set(added))
    assert added == ['G026', 'G027', 'G063', 'G097']
    assert extra == ['G058', 'G069', 'G102', 'G103']
    assert len(old) == 27 and len(incoming) == 43 and confirmed <= broad
    items = []
    for gid in added + extra:
        r = rows[gid]
        primary = groups[gid]['primary']
        source = primary['representative']
        coverage = r['coverage']
        assert r['terminal_stage'] == 'pr_filter'
        assert 10 * coverage['intersection'] > coverage['support_count']
        assert 5 * coverage['intersection'] < coverage['instance_visible_count']
        assert primary['id'] == r['mask_id'] and primary['mask_sha256'] == r['source_mask_sha256']
        mask = output / r['images']['mask']
        assert hashlib.sha256(mask.read_bytes()).hexdigest() == primary['mask_sha256']
        for image in r['images'].values():
            assert (output / image).is_file()
        items.append(dict(group_id=gid, mask_id=r['mask_id'], frame=r['frame'],
            area=r['area'], support_count=r['support_count'], coverage=coverage,
            family=source['semantic_family'], upstream_confirmed=source['semantic_hit'],
            eligibility='confirmed' if gid in added else 'broad_only',
            source_model=r['model'], source_reason=source['selection_reason'],
            semantic_top_class=r['semantic']['top_class'],
            semantic_top_fraction=r['semantic']['top_class_fraction'],
            bbox=primary['bbox'], image_size=[1200, 680], images=r['images'],
            mask_sha256=r['source_mask_sha256'], stage_decisions=r['stage_decisions']))
    report = dict(status='PASS', kind='selection_counterfactual_only', scene='room0',
        rule='Keep stages 1–4; exempt only qualifying seeds from stage 5; keep original mask pixels and IDs.',
        confirmed_rule='Primary representative semantic_hit=true and semantic_family in window/cabinet.',
        broad_rule='Stage-5 input whose SegFormer top class is windowpane/cabinet.',
        original_retained_group_ids=sorted(old), stage5_input_group_ids=sorted(incoming),
        confirmed=dict(added_group_ids=added, retained_group_ids=sorted(old | set(added)),
            remaining_pr_excluded_group_ids=sorted(removed - set(added))),
        broad=dict(added_group_ids=sorted(set(added) | set(extra)), additional_group_ids=extra,
            retained_group_ids=sorted(old | set(added) | set(extra)),
            remaining_pr_excluded_group_ids=sorted(removed - set(added) - set(extra))),
        rows=items, original_results_unchanged=True, E1_rerun=False, R2_rerun=False,
        manual_source_correction_M0466_preserved=True, source_hashes=data['source_hashes'])
    completed_root = Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019')
    if (completed_root / 'complete.json').exists():
        done = json.loads((completed_root / 'complete.json').read_text())
        assert done['status'] == 'PASS'
        chosen = json.loads((completed_root / 'v2/selection.json').read_text())
        assert chosen['retained_group_ids'] == report['broad']['retained_group_ids']
        report.update(kind='selection_preview_with_completed_broad_run', E1_rerun=True, R2_rerun=True,
            rerun_scenario='broad_35_only', confirmed_31_E1_rerun=False,
            completed_broad_run=dict(result_root=str(completed_root),
                page='../room0_auto_v2_pr35_20261019/index.html#metricPanel',
                comparison_sha256=hashlib.sha256((completed_root / 'comparison_four_way/summary.json').read_bytes()).hexdigest()))
    (output / 'pr_exemption.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
