"""Collect available frozen results; compare exactly the same R2 cohort and geometry."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
C=Path('/home/chenkejun/CVPR/experiments/replica8_swap34_20261020')
R=Path('/data/chenkejun/CVPR/results/replica8_swap34_20261020')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
KEYS=['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256',
      'profile_config_sha256','profile_semantics_sha256','evaluator_code_sha256']
SCALARS=['F1','PQ','AP','AP50','mCov','TP','FP','FN','gray_total','gray_states']
def association(root):
    p=root/'global_association.json'
    if not p.exists():return None
    a=read(p);views={x['track_id']:x for x in a['source_view_scores']}
    return dict(objects=a['objects'],source_view_scores=a['source_view_scores'],
                source_sha256=sha(p),source=str(p))
def main():
    scenes=[]
    for scene,c in read(C/'config.json').items():
        r=R/scene;oldp=Path(c['old_partial']);ne=r/'e1_repair';np_=r/'partial'
        old=read(oldp/'evaluation_r2/comparison.json');assert old['status']=='PASS'
        rows=[];evals={}
        complete=(r/'complete.json').exists() and read(r/'complete.json')['status']=='PASS'
        newp=read(np_/'evaluation_r2/comparison.json') if (np_/'evaluation_r2/comparison.json').exists() else None
        newe=read(ne/'evaluation_r2/comparison.json') if (ne/'evaluation_r2/comparison.json').exists() else None
        for stage in ['native','final']:
            base=next(x for x in old['rows'] if x['variant']=='baseline' and x['stage']==stage)
            for name,source in [('baseline','baseline'),('old_E1','old_auto_v2'),('old_partial','whole_to_partial')]:
                row=next(x for x in old['rows'] if x['variant']==source and x['stage']==stage)
                rows.append(dict(row,variant=name,selected=0 if name=='baseline' else c['old_count']))
                evals[name,stage]=oldp/'evaluation_r2'/f'{source}_{stage}'
            if newe:
                assert newe['status']=='PASS'
                ref=next(x for x in newe['rows'] if x['variant']=='before' and x['stage']==stage)
                for k in SCALARS:assert ref[k]==base[k],(scene,stage,k)
                row=next(x for x in newe['rows'] if x['variant']=='after' and x['stage']==stage)
                p=ne/'evaluation_r2'/f'after_{stage}'
                rows.append(dict(row,variant='new_E1',selected=c['new_count'],
                    owner_surface=read(p/'metrics.json')['summary']['owner_surface']))
                evals['new_E1',stage]=p
            if newp:
                assert newp['status']=='PASS'
                for variant,reference in [('baseline',base),('old_auto_v2',next(x for x in rows if x['variant']=='new_E1' and x['stage']==stage))]:
                    ref=next(x for x in newp['rows'] if x['variant']==variant and x['stage']==stage)
                    for k in SCALARS:assert ref[k]==reference[k],(scene,variant,stage,k)
                row=next(x for x in newp['rows'] if x['variant']=='whole_to_partial' and x['stage']==stage)
                rows.append(dict(row,variant='new_partial',selected=c['new_count']))
                evals['new_partial',stage]=np_/'evaluation_r2'/f'whole_to_partial_{stage}'
        locks={};surfaces={}
        for (name,stage),p in evals.items():
            ref=read(evals['baseline',stage]/'adapter_manifest.json');a=read(p/'adapter_manifest.json')
            for k in KEYS:assert a[k]==ref[k],(scene,name,stage,k)
            src=Path(a['source_surface']);assert sha(src)==a['source_surface_sha256']
            locks[name+'_'+stage]=dict(manifest_sha256=sha(p/'adapter_manifest.json'),
                surface_sha256=sha(src),path=str(p),source=str(src))
            if stage=='final':surfaces[name]=src
        classes={x['GT_ID']:x['semantic_class'] for x in read(oldp/'evaluation_r2/per_GT.json')}
        overlaps={name:dict(np.load(p/'overlap_matrix.npz')) for (name,stage),p in evals.items() if stage=='final'}
        gids=overlaps['baseline']['gt_ids'];assert len(gids)==old['GT_count']
        pergt=[]
        for gid in gids:
            q=dict(GT_ID=int(gid),semantic_class=classes[int(gid)])
            for name,z in overlaps.items():
                np.testing.assert_array_equal(z['gt_ids'],gids)
                j=int(np.flatnonzero(z['gt_ids']==gid)[0]);q[name]=float(z['iou'][:,j].max(initial=0))
            if 'new_E1' in q:q['E1_delta']=q['new_E1']-q['old_E1']
            if 'new_partial' in q:q['partial_delta']=q['new_partial']-q['old_partial']
            pergt.append(q)
        transitions={}
        data={name:dict(np.load(p)) for name,p in surfaces.items()}
        for a,b in [('baseline','new_E1'),('baseline','new_partial'),('old_E1','new_E1'),('old_partial','new_partial')]:
            if b not in data:continue
            x,y=data[a],data[b]
            np.testing.assert_array_equal(x['xyz_m'],y['xyz_m']);np.testing.assert_array_equal(x['rgb'],y['rgb'])
            xi,yi=x['instance_id'],y['instance_id']
            transitions[a+'_to_'+b]=dict(changed=int((xi!=yi).sum()),
                assigned_to_unassigned=int(((xi>0)&(yi<=0)).sum()),
                unassigned_to_assigned=int(((xi<=0)&(yi>0)).sum()),
                assigned_id_changed=int(((xi>0)&(yi>0)&(xi!=yi)).sum()))
        olda=association(Path(c['old_E1']));newa=association(ne)
        idchanges=[]
        if newa:
            def mapping(a):return {t:r for r in a['objects'] for t in r['member_track_ids']}
            before,after=mapping(olda),mapping(newa)
            for t,v in after.items():
                prev=before[t]
                if any(v[k]!=prev[k] for k in ['association','persistent_id','accepted_dice']):
                    idchanges.append(dict(track_id=t,before=prev,after=v))
        tracking=read(ne/'tracking_cache_reuse.json') if (ne/'tracking_cache_reuse.json').exists() else None
        if tracking:tracking={k:v for k,v in tracking.items() if k not in ['reused','sources']}
        statuses={}
        for n in ['status.json','failure.json','e1_repair/run_status.json','e1_repair/pipeline_status.json','partial/run_status.json','partial/progress.json']:
            if (r/n).exists():statuses[n]=read(r/n)
        d=dict(scene=scene,status='PASS' if complete else 'RUNNING',action=c['action'],
            old_count=c['old_count'],new_count=c['new_count'],GT_count=len(gids),rows=rows,per_GT=pergt,
            order=read(r/'v2/order_comparison.json'),selection=read(r/'v2/selection.json'),
            association=newa,old_association=olda,identity_changes=idchanges,tracking=tracking,
            tracking_difference_audit=read(r/'tracking_difference_audit.json') if (r/'tracking_difference_audit.json').exists() else None,
            point_changes=transitions,source_locks=locks,progress=statuses,
            source_inputs_sha256=c['frozen_input_sha256'],same_R2_geometry_GT_correspondence_verified=True)
        if complete:assert len(rows)==10
        scenes.append(d);save(r/'comparison.json',d)
    assert sum(s['GT_count'] for s in scenes)==350
    summary=[]
    for stage in ['native','final']:
        for variant in ['baseline','old_E1','new_E1','old_partial','new_partial']:
            rr=[next((x for x in s['rows'] if x['stage']==stage and x['variant']==variant),None) for s in scenes]
            if any(x is None for x in rr):continue
            row=dict(stage=stage,variant=variant,scenes=8,GT_count=350)
            for k in ['F1','PQ','AP','AP50','mCov']:row[k]=sum(x[k] for x in rr)/8
            for k in ['TP','FP','FN','gray_total']:row[k]=sum(x[k] for x in rr)
            row['gray_states']={k:sum(x['gray_states'][k] for x in rr) for k in ['U','T','C']}
            row['micro_F1']=2*row['TP']/(2*row['TP']+row['FP']+row['FN'])
            summary.append(row)
    result=dict(status='PASS' if all(s['status']=='PASS' for s in scenes) else 'RUNNING',
        protocol='object_observed_repair / revision 2',frozen=False,scoring_source='cd260569e89de1e9525f82c45ee10842b80616b0',
        GT_count=350,old_total=201,new_total=196,scenes=scenes,summary=summary,
        aggregate_policy='F1/PQ/AP/AP50/mCov: arithmetic scene mean over exactly 8 scenes. TP/FP/FN and gray points: sum. micro_F1 shown separately.',
        historical_room2_exceptions_preserved=['G017','G025'])
    save(R/'comparison.json',result)
    print(json.dumps(dict(status=result['status'],completed=[s['scene'] for s in scenes if s['status']=='PASS'],
        summary=[x for x in summary if x['stage']=='final']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
