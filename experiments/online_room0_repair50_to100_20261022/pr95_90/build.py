#!/usr/bin/env python3
"""Re-evaluate the frozen A/B 100-frame measurements using the approved strict 95/90 skip rule.
Only this independent result tree is written. Original candidates, maps, and tracking are read-only.
"""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, time
from fractions import Fraction
import numpy as np
from PIL import Image

C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022')
T=Path('/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022')
CODE=C/'pr95_90'
OUT=T/'pr95_90'
WEB=T/'review/pr95-90'
OLD_RULE='p >= 0.95 and r >= 0.95'
NEW_RULE='(p > 0.95 and r > 0.90) or (r > 0.95 and p > 0.90)'
INT_RULE='(20*c > 19*n and 10*c > 9*i) or (20*c > 19*i and 10*c > 9*n)'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,data):
    path=Path(path)
    assert path.is_relative_to(OUT) or path.is_relative_to(WEB) or path.is_relative_to(CODE)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
def link(target,path):
    assert path.is_relative_to(OUT) or path.is_relative_to(WEB)
    if path.is_symlink():assert path.resolve()==target.resolve()
    else:
        assert not path.exists()
        path.symlink_to(target)
def fns(src):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}

def setup_rules():
    original=C/'prefix100/v2/filter_rules.py'
    old_source=original.read_text()
    old='return i > 0 and 20 * c >= 19 * n and 20 * c >= 19 * i'
    new='return i > 0 and ((20 * c > 19 * n and 10 * c > 9 * i) or\n                      (20 * c > 19 * i and 10 * c > 9 * n))'
    assert old_source.count(old)==1
    new_source=old_source.replace(old,new)
    assert [k for k in fns(old_source) if fns(old_source)[k]!=fns(new_source)[k]]==['skip_repair']
    (CODE/'filter_rules.py').write_text(new_source)
    old_mod=module('frozen_rules',original)
    new_mod=module('pr95_90_rules',CODE/'filter_rules.py')
    sem=module('frozen_semantic_exemption',C/'prefix100/v2/semantic_exemption.py')
    # Exact strict boundary truth-table, including exchanged p/r and old equality.
    cases=[(95,100,100,False),(90,100,90,False),(90,90,100,False),
           (96,100,106,True),(96,106,100,True),(95,100,99,True),
           (95,99,100,True),(0,100,0,False),(90,100,100,False),(99,100,100,True)]
    for c,n,i,expected in cases:
        assert new_mod.skip_repair(c,n,i)==expected
        rational=i>0 and ((Fraction(c,n)>Fraction(95,100) and Fraction(c,i)>Fraction(90,100)) or
                        (Fraction(c,i)>Fraction(95,100) and Fraction(c,n)>Fraction(90,100)))
        assert new_mod.skip_repair(c,n,i)==rational
    return old_mod,new_mod,sem,dict(changed_functions=['skip_repair'],
        all_other_functions_ast_identical=True,strict_boundary_cases=len(cases),
        old_filter_sha256=sha(original),new_filter_sha256=sha(CODE/'filter_rules.py'),
        semantic_exemption_sha256=sha(C/'prefix100/v2/semantic_exemption.py'))

def validate_members(O,grouping,selected,lower):
    members={}
    for group in grouping['groups']:
        if group['id'] in selected:
            for idx in group['members']:
                seed=grouping['seeds'][idx]
                members[seed['id']]=seed
    alias_count=0
    rows=[]
    for mid,seed in sorted(members.items()):
        path=O/'v1'/seed['mask']
        assert sha(path)==seed['mask_sha256']
        arr=np.asarray(Image.open(path))>0
        assert arr.shape==(680,1200) and int(arr.sum())==seed['area']
        edge=dict(left=int(arr[:,0].sum()),right=int(arr[:,-1].sum()),
                  top=int(arr[0,:].sum()),bottom=int(arr[-1,:].sum()))
        assert sum(edge.values())==0 and not seed['touches_any_edge']
        assert lower<=seed['frame']<=495 and seed['frame']%5==0
        crops=[]
        for alias in seed['aliases']:
            alias_count+=1
            assert lower<=alias['frame']<=495 and alias['frame']%5==0
            x0,y0,x1,y1=alias['source_box']
            contacts={}
            if x0>0:contacts['left']=int(arr[y0:y1,x0].sum())
            if x1<1200:contacts['right']=int(arr[y0:y1,x1-1].sum())
            if y0>0:contacts['top']=int(arr[y0,x0:x1].sum())
            if y1<680:contacts['bottom']=int(arr[y1-1,x0:x1].sum())
            assert sum(contacts.values())==0
            assert not alias['touches_crop'] and not alias['touches_frame']
            crops.append(dict(source_box=alias['source_box'],edge_pixels=contacts))
        rows.append(dict(mask_id=mid,frame=seed['frame'],mask_sha256=seed['mask_sha256'],
                         full_frame_edge_pixels=edge,source_crop_checks=crops))
    return dict(status='PASS',unique_primary_and_auxiliary_masks=len(members),
                primary_masks=len(selected),auxiliary_masks=len(members)-len(selected),
                source_aliases=alias_count,full_frame_contacts=0,artificial_crop_contacts=0,
                source_raw_frame_range=[lower,495],actual_member_raw_frames=sorted({s['frame'] for s in members.values()}),
                records=rows)

def reason(row):
    if row['new_retained']:
        if row['stage5_restored_by_semantics']:return '新规则保留；第5步由背景语义窗/柜豁免恢复'
        return '新规则保留；未命中第4步95/90排除，且通过后续筛选'
    if row['new_terminal_stage']=='skip_repair_B':
        return '第4步排除：'+('p > 95% 且 r > 90%' if row['p_gt95_and_r_gt90'] else 'r > 95% 且 p > 90%')
    return {'semantic':'背景语义排除','covisible_iou_only':'共视去重排除',
            'full_support_containment':'直接包含排除','pr_filter':'第5步 p > 10% 且 r < 20%，无适用豁免'}.get(row['new_terminal_stage'],row['new_terminal_stage'])

def build_arm(arm,directory,old_mod,new_mod,sem_mod,code_audit):
    started=time.perf_counter()
    O=T/directory/'all_edge_20261021'
    dest=OUT/arm
    page=WEB/arm
    dest.mkdir(parents=True,exist_ok=True);page.mkdir(parents=True,exist_ok=True)
    files=[O/'v2/measurements.json',O/'v2/selection.json',O/'v2/seed_manifest.json',
           O/'v2/semantic/classification.json',O/'v1/seed_manifest.json',O/'v1/grouping.json',
           O/'review/data.json',O/'boundary_audit.json',
           T/'tracking100'/arm/'input_freeze.json']
    before={str(p):sha(p) for p in files}
    measurements=read(files[0]);previous=read(files[1]);previous_manifest=read(files[2])
    semantic=read(files[3]);full_manifest=read(files[4]);grouping=read(files[5]);source_page=read(files[6])
    old=sem_mod.apply_semantic_exemption(old_mod.run_filters(copy.deepcopy(measurements),allow_chained_containment=True),semantic)
    assert old=={k:previous[k] for k in old}, 'Frozen old selection replay mismatch'
    new=sem_mod.apply_semantic_exemption(new_mod.run_filters(copy.deepcopy(measurements),allow_chained_containment=True),semantic)
    assert new['stages'][:3]==old['stages'][:3]
    old_set=set(old['retained_group_ids']);new_set=set(new['retained_group_ids'])
    records={r['group_id']:r for r in new['records']}
    old_records={r['group_id']:r for r in old['records']}
    coverage={r['group_id']:r for r in measurements['coverage']}
    semantic_by_id={r['group_id']:r for r in semantic['records']}
    full_by_id={g['id']:g for g in full_manifest['groups']}
    group_by_id={g['id']:g for g in grouping['groups']}
    lower=0 if arm=='A' else 250
    source_policy='all_observed_mapping_frames_1_to100' if arm=='A' else 'new_mapping_frames_51_to100'
    source_range=[1 if arm=='A' else 51,100]
    meta=dict(arm=arm,rule_version='pr95_90',candidate_source_policy=source_policy,
        candidate_raw_frame_range=[lower,495],candidate_processed_frame_range=source_range,
        scoring_history=[1,100],scoring_raw_frame_range=[0,495],processed_frames=100,raw_cutoff=495,
        repair50_applied=True,repair100_applied=False,model_inference_performed=False,
        tracking_requested=False,tracking_performed=False,map_write=False,GT_used=False,
        source_result_directory=str(O))
    comparison_rows=[]
    for gid in sorted(records):
        cov=coverage.get(gid)
        s=semantic_by_id[gid]
        group=full_by_id[gid]
        group_seeds=[grouping['seeds'][idx] for idx in group_by_id[gid]['members']]
        hit_aliases=[a for gs in group_seeds for a in gs['aliases'] if a['semantic_hit']]
        primary=group['primary']
        c,n,i=([cov[k] for k in ['intersection','support_count','instance_visible_count']] if cov else [None]*3)
        a=bool(cov and i>0 and 20*c>19*n and 10*c>9*i)
        b=bool(cov and i>0 and 20*c>19*i and 10*c>9*n)
        if cov:assert new_mod.skip_repair(c,n,i)==(a or b)
        row=dict(group_id=gid,mask_id=primary['id'],frame=primary['frame'],
            processed_frame=primary['frame']//5+1,mask_sha256=primary['mask_sha256'],
            p=cov['p'] if cov else None,r=cov['r'] if cov else None,
            intersection=c,support_count=n,instance_visible_count=i,instance_id=cov['instance_id'] if cov else None,
            old_retained=gid in old_set,new_retained=gid in new_set,
            newly_excluded=gid in old_set-new_set,newly_retained=gid in new_set-old_set,
            old_terminal_stage=old_records[gid]['terminal_stage'],new_terminal_stage=records[gid]['terminal_stage'],
            p_gt95_and_r_gt90=a,r_gt95_and_p_gt90=b,
            stage4_old_excluded=gid in old['stages'][3]['removed_group_ids'],
            stage4_new_excluded=gid in new['stages'][3]['removed_group_ids'],
            stage5_semantic_class=s['top_class'],
            stage5_semantic_eligible=s['top_class'] in ['windowpane','cabinet'],
            stage5_semantic_exemption_in_scope=gid in new['semantic_pr_exempt_group_ids'],
            stage5_restored_by_semantics=gid in new['added_group_ids'],
            stage5_exemption_scope='stage5 p>0.10 and r<0.20 only; never restores stage4 exclusions',
            frontend_semantic_hit=bool(hit_aliases),
            frontend_target_families=sorted({a['semantic_family'] for a in hit_aliases}),
            source_model=primary['representative']['model'])
        row['reason_zh']=reason(row)
        comparison_rows.append(row)
    source_metadata_corrections=[]
    if full_manifest['rule'].get('candidate_raw_frame_range')!=[lower,495]:
        source_metadata_corrections.append(dict(field='rule.candidate_raw_frame_range',
             source=full_manifest['rule'].get('candidate_raw_frame_range'),new=[lower,495],numeric_inputs_changed=False))
    if full_manifest['rule'].get('candidate_processed_frame_range')!=source_range:
        source_metadata_corrections.append(dict(field='rule.candidate_processed_frame_range',
             source=full_manifest['rule'].get('candidate_processed_frame_range'),new=source_range,numeric_inputs_changed=False))
    selection=copy.deepcopy(new)
    selection.update(meta,case_id='room0_prefix100_'+arm+'_pr95_90',
        source_case_id=measurements['case_id'],
        selection_variant='stage4_strict_pr95_90_with_original_stage5_semantic_exemption',
        skip_repair_rule=dict(old=OLD_RULE,new=NEW_RULE,integer=INT_RULE,strict=True),
        semantic_exemption_scope='stage5 only; stages 1-4 are never restored',
        old_retained_group_ids=old['retained_group_ids'],
        newly_excluded_group_ids=sorted(old_set-new_set),newly_retained_group_ids=sorted(new_set-old_set))
    manifest=copy.deepcopy(full_manifest)
    manifest.update(meta,status='complete',mask_source_directory=str(O/'v1'),
                    groups=[copy.deepcopy(full_by_id[g]) for g in sorted(new_set)],
                    selection_path=str(dest/'selection.json'),retained_count=len(new_set))
    manifest['rule'].update(candidate_source_policy=source_policy,
        candidate_raw_frame_range=[lower,495],candidate_processed_frame_range=source_range,
        skip_repair=NEW_RULE,skip_repair_strict=True,
        stage5_pr='p>0.10 and r<0.20',stage5_semantic_exemption=['windowpane','cabinet'],
        semantic_exemption_applies_only_to_stage=5,rule_version='pr95_90')
    previous_by_id={g['id']:g for g in previous_manifest['groups']}
    for g in manifest['groups']:
        assert g==full_by_id[g['id']]
        if g['id'] in previous_by_id:assert g==previous_by_id[g['id']]
    boundary=validate_members(O,grouping,new_set,lower)
    equality=[r['group_id'] for r in measurements['coverage'] if
        20*r['intersection'] in [19*r['support_count'],19*r['instance_visible_count']] or
        10*r['intersection'] in [9*r['support_count'],9*r['instance_visible_count']]]
    comparison=dict(**meta,status='PASS',old_rule=OLD_RULE,new_rule=NEW_RULE,integer_rule=INT_RULE,
        old_retained=len(old_set),new_retained=len(new_set),
        old_retained_group_ids=sorted(old_set),new_retained_group_ids=sorted(new_set),
        newly_excluded_group_ids=sorted(old_set-new_set),newly_retained_group_ids=sorted(new_set-old_set),
        old_stages=old['stages'],new_stages=new['stages'],
        stage5_semantic_exempt_group_ids=new['semantic_pr_exempt_group_ids'],
        stage5_actually_restored_group_ids=new['added_group_ids'],
        exact_90_or_95_equality_group_ids=equality,rows=comparison_rows,
        old_selection_path=str(O/'v2/selection.json'),old_manifest_path=str(O/'v2/seed_manifest.json'),
        new_selection_path=str(dest/'selection.json'),new_manifest_path=str(dest/'seed_manifest.json'))
    page_data=dict(**meta,old_rule=OLD_RULE,new_rule=NEW_RULE,
        summary=dict(old_retained=len(old_set),retained=len(new_set),newly_excluded=len(old_set-new_set),
            newly_retained=len(new_set-old_set),initial=source_page['summary']['initial'],stages=new['stages'],
            retained_group_ids=sorted(new_set),retained_primary_and_auxiliary_masks=boundary['unique_primary_and_auxiliary_masks'],
            retained_frame_touch=0,retained_crop_touch=0,
            stage5_exempt_group_ids=new['semantic_pr_exempt_group_ids'],
            stage5_actually_restored_group_ids=new['added_group_ids']),
        rows=[])
    source_rows={r['group']:r for r in source_page['rows'] if r['primary']}
    for cr in comparison_rows:
        row=copy.deepcopy(source_rows[cr['group_id']])
        row.update(cr)
        row.update(status='kept' if cr['new_retained'] else 'newly_excluded' if cr['newly_excluded'] else 'excluded',
                   reason=cr['reason_zh'])
        page_data['rows'].append(row)
    after={str(p):sha(p) for p in files}
    assert before==after
    audit=dict(**meta,status='PASS',old_rules_reproduce_frozen_selection_exactly=True,
        stages1_to3_identical=True,only_stage4_skip_condition_changed=True,
        stage5_pr_rule_and_semantic_exemption_unchanged=True,old_inputs_byte_identical=True,
        primary_and_auxiliary_masks_byte_identical=True,strict_threshold_boundary_tests=code_audit,
        source_hashes=before,source_metadata_corrections_only_in_new_output=source_metadata_corrections,
        retained_boundary_audit='boundary_audit.json',retained_primary_masks=len(new_set),
        retained_auxiliary_masks=boundary['auxiliary_masks'],elapsed_seconds=time.perf_counter()-started)
    for name,value in [('selection.json',selection),('seed_manifest.json',manifest),
                       ('comparison.json',comparison),('boundary_audit.json',boundary),
                       ('audit.json',audit),('page_data.json',page_data)]:
        dump(dest/name,value)
    for rel in ['measurements.json','classification.json']:
        link(O/('v2/measurements.json' if rel=='measurements.json' else 'v2/semantic/classification.json'),dest/rel)
    for rel in ['assets','masks','frames']:link(O/'review'/rel,page/rel)
    for rel,target in [('data.json','page_data.json'),('selection.json','selection.json'),
                       ('seed_manifest.json','seed_manifest.json'),('comparison.json','comparison.json'),
                       ('audit.json','audit.json'),('boundary_audit.json','boundary_audit.json')]:
        link(dest/target,page/rel)
    return dict(arm=arm,old_retained=len(old_set),new_retained=len(new_set),
        retained_ids=sorted(new_set),newly_excluded_ids=sorted(old_set-new_set),
        stage5_actual_restores=new['added_group_ids'],retained_primary_and_aux_masks=boundary['unique_primary_and_auxiliary_masks'],
        source_primary_historical=sum(g['primary']['frame']<250 for g in manifest['groups']),
        source_primary_new=sum(g['primary']['frame']>=250 for g in manifest['groups']),
        result_directory=str(dest),page_url='http://127.0.0.1:8798/online-repair/pr95-90/'+arm+'/',
        seed_manifest_sha256=sha(dest/'seed_manifest.json'))

def main():
    start=time.perf_counter()
    CODE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    old,new,sem,code_audit=setup_rules()
    locks=read(T/'tracking100/A/input_freeze.json')['original_baseline_source_hashes']
    locks.update(read(T/'prefix100_new50/arm_provenance.json')['shared_sources_sha256'])
    committed=read(T/'committed50.json')
    for field in ['checkpoint','audit','final_map']:locks[committed[field]]=committed[field+'_sha256']
    assert all(sha(p)==h for p,h in locks.items())
    summaries=[build_arm('A','prefix100',old,new,sem,code_audit),
               build_arm('B','prefix100_new50',old,new,sem,code_audit)]
    assert all(sha(p)==h for p,h in locks.items())
    records=[read(OUT/arm/'seed_manifest.json')['groups'] for arm in ['A','B']]
    keys=[{(g['primary']['frame'],g['primary']['mask_sha256']):g for g in groups} for groups in records]
    common=[]
    for key in sorted(keys[0].keys()&keys[1].keys()):
        common.append(dict(frame=key[0],mask_sha256=key[1],A_group=keys[0][key]['id'],B_group=keys[1][key]['id']))
    report=dict(status='PASS',rule_version='pr95_90',old_rule=OLD_RULE,new_rule=NEW_RULE,
        scope='only stage4 skip_repair condition changed; original stage5 semantic exemption preserved',
        arms=summaries,common_retained_primary_seeds=common,common_seed_count=len(common),
        original_maps_and_committed50_unchanged=True,source_hashes=locks,
        model_inference_performed=False,tracking_performed=False,repair100_applied=False,
        total_selection_and_validation_seconds=time.perf_counter()-start,code_audit=code_audit)
    dump(OUT/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},ensure_ascii=False))
if __name__=='__main__':main()
