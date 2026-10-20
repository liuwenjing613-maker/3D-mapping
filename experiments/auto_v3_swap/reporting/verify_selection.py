"""Cross-check the experiment against frozen historical decisions and masks."""
from pathlib import Path
import copy,hashlib,json,runpy
W=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
SRC=W/'worktrees/auto-v2/experiments/auto_v2'
P=W/'results/固定案例_三模型对比_20261006/replica8_auto_E1_20261019'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=(SRC/'replica6_20261019/room1/v2/filter_rules.py').read_text(encoding='utf-8')
changed=(SRC/'replica8_swap34_20261020/rules/filter_rules.py').read_text(encoding='utf-8')
block="""    current = finish_stage('skip_repair_B', current,
                           [gid for gid in current if skip_repair(*args(gid))], coverage)
"""
assert original.replace(block,'').split()==changed.replace(block,'').split()
oldrun=runpy.run_path(str(SRC/'replica6_20261019/room1/v2/filter_rules.py'))['run_filters']
newrun=runpy.run_path(str(SRC/'replica8_swap34_20261020/rules/filter_rules.py'))['run_filters']
exempt=runpy.run_path(str(SRC/'replica8_swap34_20261020/rules/semantic_exemption.py'))['apply_semantic_exemption']
result=[]
for scene in read(O/'delivery/comparison.json')['scenes']:
    name=scene['scene'];p=P/'filtering'/name/'reports';old=read(p/'selection.json')
    m=read(p/'measurements.json');sem=read(p/'semantic.json');rawold=oldrun(m);oldex=exempt(rawold,sem)
    expected=set(oldex['retained_group_ids'])|set(old['manual_restored_group_ids'])
    assert expected==set(old['retained_group_ids'])
    new=newrun(m);newex=exempt(new,sem);expected=set(newex['retained_group_ids'])|set(old['manual_restored_group_ids'])
    actual=scene['selection'];assert expected==set(actual['retained_group_ids'])
    assert old['stages'][:2]==actual['stages'][:2]
    old_records={r['group_id']:r for r in old['records']};new_records={r['group_id']:r for r in actual['records']}
    for gid,r in old_records.items():
        for k in ['group_id','mask_id','frame','mask_sha256']:assert r[k]==new_records[gid][k]
    assert len(old_records)==len(new_records)
    assert all(new_records[g]['decision']=='retain' for g in actual['semantic_pr_exempt_group_ids'])
    assert actual['semantic_pr_exempt_group_ids']==oldex['semantic_pr_exempt_group_ids']
    removed=[]
    for gid in scene['order']['removed']:
        row=new_records[gid];assert row['terminal_stage']=='full_support_containment'
        witnesses=row['stage_decisions'][-1]['evidence']
        for witness in witnesses:
            ref=witness['larger_group_id'];assert old_records[ref]['terminal_stage']=='skip_repair_B'
        removed.append(dict(group_id=gid,mask_id=row['mask_id'],witnesses=witnesses))
    result.append(dict(scene=name,old=scene['old_count'],new=scene['new_count'],v1_count=len(new_records),
        removed=removed,old_filter_reproduced=True,unchanged_stage_1_2=True,semantic_exemption_unchanged=True,
        all_source_mask_frame_ID_hashes_unchanged=True))
out=dict(status='PASS',only_code_change_is_stage_order=True,scenes=result,total_v1=sum(r['v1_count'] for r in result),total_selected=sum(r['new'] for r in result))
assert out['total_v1']==637 and out['total_selected']==196
(O/'selection_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
