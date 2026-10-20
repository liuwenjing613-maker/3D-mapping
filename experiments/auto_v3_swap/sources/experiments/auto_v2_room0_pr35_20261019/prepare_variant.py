"""Freeze the approved 35-seed selection without modifying the 27-seed run."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
from PIL import Image
C=Path(__file__).resolve().parent
OLD=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019')
R=Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    assert not (R/'v2/complete.json').exists()
    assert read(OLD/'e1_repair/run_complete.json')['status']=='PASS'
    frozen=read(OLD/'e1_repair/pipeline_code_freeze.json')
    for p,h in frozen['files'].items():assert sha(Path(p))==h,p
    locks={str(p):sha(p) for p in [OLD/'v1/seed_manifest.json',OLD/'v2/selection.json',OLD/'v2/seed_manifest.json',OLD/'v2/measurements.json',OLD/'v2/semantic/classification.json',OLD/'comparison_three_way/summary.json']}
    olddone=read(OLD/'v2/complete.json');assert sha(OLD/'v2/selection.json')==olddone['selection_sha256']
    assert sha(OLD/'v2/seed_manifest.json')==olddone['manifest_sha256']
    original=read(OLD/'v2/selection.json');source=read(OLD/'v1/seed_manifest.json');semantic=read(OLD/'v2/semantic/classification.json')
    assert semantic['source_manifest_sha256']==sha(OLD/'v1/seed_manifest.json')
    sem={r['group_id']:r for r in semantic['records']}
    incoming=set(original['stages'][3]['retained_group_ids']);before=set(original['retained_group_ids'])
    exempt={g for g in incoming if sem[g]['top_class'] in ['windowpane','cabinet']}
    added=sorted(exempt-before)
    assert added==['G026','G027','G058','G063','G069','G097','G102','G103']
    retained=sorted(before|exempt);assert len(retained)==35
    report=read(OLD/'review/pr_exemption.json');assert report['broad']['retained_group_ids']==retained
    selection=copy.deepcopy(original)
    selection.update(case_id='room0_auto_v2_pr35_20261019',retained_group_ids=retained,automatic_retained_group_ids=retained,
        semantic_pr_exempt_group_ids=sorted(exempt),added_group_ids=added,
        original_pr_excluded_group_ids=original['final_pr_excluded_group_ids'],
        final_pr_excluded_group_ids=sorted(set(original['final_pr_excluded_group_ids'])-exempt),
        selection_variant='stage5_SegFormer_windowpane_cabinet_exemption',manual_restored_group_ids=[])
    final_stage=selection['stages'][4]
    final_stage.update(retained=35,removed=8,retained_group_ids=retained,
        removed_group_ids=selection['final_pr_excluded_group_ids'],semantic_exempt_group_ids=sorted(exempt))
    for row in selection['records']:
        gid=row['group_id']
        if gid not in exempt:continue
        decision=row['stage_decisions'][-1];assert decision['stage']=='pr_filter'
        decision['decision']='retain';decision['evidence']['semantic_exemption']=dict(class_name=sem[gid]['top_class'],rule='SegFormer top class windowpane/cabinet; exempt stage 5 only')
        if gid in added:row.update(decision='retain',terminal_stage='semantic_pr_exemption')
    assert selection['stages'][:4]==original['stages'][:4]
    assert set(selection['final_pr_excluded_group_ids'])==incoming-set(retained)
    chosen=copy.deepcopy(source);chosen['groups']=[]
    for tid,g in enumerate(source['groups'],1):
        if g['id'] not in retained:continue
        p=g['primary'];path=OLD/'v1'/p['mask'];assert sha(path)==p['mask_sha256']
        mask=np.asarray(Image.open(path))>0;assert int(mask.sum())==p['area']
        assert hashlib.sha256(p['frame'].to_bytes(4,'little')+np.packbits(mask.ravel()).tobytes()).hexdigest()==p['digest']
        chosen['groups'].append(dict(copy.deepcopy(g),original_track_id=tid))
    correction=next(g for g in chosen['groups'] if g['id']=='G030')['primary']
    assert correction['id']=='M0466' and correction['representative']['model']==0 and 'G030' not in exempt
    assert correction['mask_sha256']=='aa312039fa7b94998f7807f5e39f3557aa152366eba35d121bae853174c2471c'
    for name in ['v2','e1_repair']:(R/name).mkdir(exist_ok=True)
    (R/'v1').symlink_to(OLD/'v1',target_is_directory=True)
    (R/'v2/semantic').symlink_to(OLD/'v2/semantic',target_is_directory=True)
    (R/'v2/measurements.json').symlink_to(OLD/'v2/measurements.json')
    (R/'v2/masks').symlink_to(OLD/'v1/masks',target_is_directory=True)
    save(R/'v2/selection.json',selection)
    chosen.update(status='automatic_v2_pr35_complete',scene='room0',mask_source_directory=str(OLD/'v1'),original_manifest_sha256=sha(OLD/'v1/seed_manifest.json'),selection_sha256=sha(R/'v2/selection.json'),manual_overrides=[],GT_used=False)
    save(R/'v2/seed_manifest.json',chosen)
    save(R/'v2/complete.json',dict(status='PASS',input=103,retained=35,stages=selection['stages'],manual_overrides=[],selection_sha256=sha(R/'v2/selection.json'),manifest_sha256=sha(R/'v2/seed_manifest.json')))
    for p,h in locks.items():assert sha(Path(p))==h,p
    save(R/'v2/source_lock.json',dict(status='PASS',source_sha256=locks,code_sha256=sha(Path(__file__)),original_sources_unchanged=True,first_four_stages_identical=True,original_27_subset_of_35=True,all_masks_and_original_track_ids_preserved=True,M0466_CropFormer_preserved=True,added=added,GT_used=False))
    print('SELECTION_FROZEN',json.dumps(dict(before=27,after=35,added=added,stage5_exempt_count=len(exempt))),flush=True)
if __name__=='__main__':main()
