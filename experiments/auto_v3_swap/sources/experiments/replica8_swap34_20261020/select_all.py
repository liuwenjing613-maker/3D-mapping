"""Replay every scene from frozen measurements and preserve historical exceptions."""
from pathlib import Path
import copy, hashlib, json, sys
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
sys.path.insert(0,str(C/'rules'))
from filter_rules import run_filters
from semantic_exemption import apply_semantic_exemption
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main():
    config=read(C/'config.json')
    for scene,c in config.items():
        r=R/scene;r.mkdir(exist_ok=True);v=r/'v2';v.mkdir(exist_ok=True)
        for p,h in c['frozen_input_sha256'].items():assert sha(p)==h,p
        old=read(c['selection']);manifest=read(Path(c['v1'])/'seed_manifest.json')
        raw=run_filters(read(c['measurements']))
        new=apply_semantic_exemption(raw,read(Path(c['semantic'])/'classification.json'))
        assert new['stages'][:2]==old['stages'][:2]
        if scene=='room2':
            assert old['manual_restored_group_ids']==['G017','G025']
            for gid in old['manual_restored_group_ids']:
                assert gid in new['final_pr_excluded_group_ids']
                rec=next(x for x in new['records'] if x['group_id']==gid)
                prev=next(x for x in old['records'] if x['group_id']==gid)
                assert rec['terminal_stage']=='pr_filter'
                rec.update(decision='retain',terminal_stage='manual_room2_override')
                rec['stage_decisions'].append(copy.deepcopy(prev['stage_decisions'][-1]))
            new['manual_restored_group_ids']=old['manual_restored_group_ids']
            new['retained_group_ids']=sorted(set(new['retained_group_ids'])|set(old['manual_restored_group_ids']))
            new['final_pr_excluded_group_ids']=sorted(set(new['final_pr_excluded_group_ids'])-set(old['manual_restored_group_ids']))
            new['historical_exception_note']='G017/G025 retained exactly as the old room2 run; not a new semantic exemption.'
            new['stages'][-1].update(retained=len(new['retained_group_ids']),
                retained_group_ids=new['retained_group_ids'],removed=len(new['final_pr_excluded_group_ids']),
                removed_group_ids=new['final_pr_excluded_group_ids'],manual_restored_group_ids=old['manual_restored_group_ids'])
        new['selection_variant']='containment_before_coverage_with_unchanged_exceptions'
        kept=set(new['retained_group_ids']);before=set(old['retained_group_ids'])
        assert len(kept)==c['new_count']
        save(v/'selection_without_exemption.json',raw);save(v/'selection.json',new)
        for dest,src in [('v1',c['v1']),('v2/semantic',c['semantic']),('v2/measurements.json',c['measurements']),('v2/masks',c['v1']+'/masks')]:
            (r/dest).symlink_to(src,target_is_directory=not dest.endswith('.json'))
        selected=copy.deepcopy(manifest)
        selected['groups']=[dict(copy.deepcopy(g),original_track_id=i) for i,g in enumerate(manifest['groups'],1) if g['id'] in kept]
        selected.update(status='automatic_v2_complete',scene=scene,mask_source_directory=c['v1'],
            original_manifest_sha256=sha(Path(c['v1'])/'seed_manifest.json'),selection_sha256=sha(v/'selection.json'),
            manual_overrides=new['manual_restored_group_ids'],GT_used=False)
        save(v/'seed_manifest.json',selected)
        compare=dict(status='PASS',scene=scene,old_count=len(before),new_count=len(kept),
            added=sorted(kept-before),removed=sorted(before-kept),old_stages=old['stages'],new_stages=new['stages'],
            measurements_sha256=sha(c['measurements']),source_measurements=c['measurements'],
            original_selection_sha256=sha(c['selection']),only_filter_order_changed=True,
            semantic_stage5_exemption_unchanged=True,historical_manual_exceptions_preserved=new['manual_restored_group_ids'],
            action=c['action'],GT_used=False)
        save(v/'order_comparison.json',compare)
        save(v/'source_lock.json',dict(source_sha256=c['frozen_input_sha256'],code_sha256={p.name:sha(p) for p in (C/'rules').glob('*.py')}))
        save(v/'complete.json',dict(status='PASS',input=len(manifest['groups']),retained=len(kept),stages=new['stages'],
            selection_sha256=sha(v/'selection.json'),manifest_sha256=sha(v/'seed_manifest.json')))
        if c['action']!='rerun':
            actual=read(Path(c['reuse_E1'])/'confirmed_selection.json')
            inc={x['group_id']:x for x in actual['included']}
            assert kept==set(inc)
            for i,g in enumerate(manifest['groups'],1):
                if g['id'] not in kept:continue
                row=inc[g['id']];p=g['primary']
                assert (i,p['id'],p['frame'],p['mask_sha256'])==(row['original_track_id'],row['mask_id'],row['frame'],row['mask_sha256'])
                assert sha(Path(c['v1'])/p['mask'])==p['mask_sha256']
            for dest,src in [('e1_repair',c['reuse_E1']),('partial',c['reuse_partial'])]:
                assert read(Path(src)/'evaluation_r2/comparison.json')['status']=='PASS'
                (r/dest).symlink_to(src,target_is_directory=True)
            if c['action']=='reuse_identical_selection':assert kept==before
            save(r/'reuse_audit.json',dict(status='PASS',action=c['action'],seed_masks_frames_track_ids_identical=True,
                e1_source=c['reuse_E1'],partial_source=c['reuse_partial'],
                selected_input_sha256=sha(Path(c['reuse_E1'])/'confirmed_selection.json')))
            save(r/'complete.json',dict(status='PASS',execution=c['action']))
        else:
            (r/'e1_repair').mkdir();(r/'partial').mkdir()
        print(json.dumps(compare),flush=True)
    save(R/'selection_complete.json',dict(status='PASS',scenes=8,total_retained=sum(d['new_count'] for d in config.values())))
if __name__=='__main__':main()
