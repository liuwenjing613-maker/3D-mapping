"""Use the approved room0 stage-5 exemption, preserving stages 1--4."""
from copy import deepcopy

def apply_semantic_exemption(original,semantic):
    assert not original['manual_restored_group_ids']
    sem={r['group_id']:r for r in semantic['records']}
    incoming=set(original['stages'][3]['retained_group_ids'])
    before=set(original['retained_group_ids'])
    exempt={g for g in incoming if sem[g]['top_class'] in ['windowpane','cabinet']}
    retained=sorted(before|exempt);added=sorted(exempt-before)
    selected=deepcopy(original)
    selected.update(retained_group_ids=retained,automatic_retained_group_ids=retained,
        semantic_pr_exempt_group_ids=sorted(exempt),added_group_ids=added,
        original_pr_excluded_group_ids=original['final_pr_excluded_group_ids'],
        final_pr_excluded_group_ids=sorted(incoming-set(retained)),
        selection_variant='stage5_SegFormer_windowpane_cabinet_exemption',manual_restored_group_ids=[])
    selected['stages'][4].update(retained=len(retained),removed=len(incoming)-len(retained),
        retained_group_ids=retained,removed_group_ids=selected['final_pr_excluded_group_ids'],
        semantic_exempt_group_ids=sorted(exempt))
    for row in selected['records']:
        gid=row['group_id']
        if gid not in exempt:continue
        decision=row['stage_decisions'][-1];assert decision['stage']=='pr_filter'
        decision['decision']='retain'
        decision['evidence']['semantic_exemption']=dict(class_name=sem[gid]['top_class'],rule='SegFormer top class windowpane/cabinet; exempt stage 5 only')
        if gid in added:row.update(decision='retain',terminal_stage='semantic_pr_exemption')
    assert selected['stages'][:4]==original['stages'][:4]
    assert before<=set(retained)<=incoming
    return selected
