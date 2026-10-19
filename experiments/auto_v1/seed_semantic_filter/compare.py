"""Compare only frozen semantic decisions to frozen manual and CLIP references."""
from pathlib import Path
import argparse
import json
import shutil
import zipfile
from run import sha, save

STRUCTURE_AUDIT=['G001','G002','G003','G004','G005','G008','G011','G014','G023','G053']
RUG_AUDIT=['G016','G019','G024','G034','G044','G059']
PROTECTED_AUDIT=['G006','G010','G013','G040']


def compare(result,human,clip):
    if human['status']!='confirmed' or human['qa_only']:
        raise ValueError('Expected confirmed production human selection')
    if not result['source_manifest_sha256']==human['manifest_sha256']==clip['source_manifest_sha256']:
        raise ValueError('References use different original masks')
    for r in [result,clip]:
        if r['human_selection_used'] or r['GT_used']: raise ValueError('Automatic result used comparison labels')
    included={r['group_id']:r for r in human['included']}; excluded={r['group_id']:r for r in human['excluded']}
    automatic={r['group_id']:r for r in result['records']}; old={r['group_id']:r for r in clip['records']}
    if len(included)+len(excluded)!=len(human['included'])+len(human['excluded']) or set(included)&set(excluded):
        raise ValueError('Invalid human identity partition')
    if len(automatic)!=len(result['records']) or len(old)!=len(clip['records']): raise ValueError('Duplicate automatic groups')
    if not (set(included)|set(excluded))==set(automatic)==set(old): raise ValueError('Incomplete comparison')
    groups={k:[] for k in ['both_excluded','auto_only_excluded','human_only_excluded','both_retained']}
    versus_clip={k:[] for k in ['both_excluded','semantic_only_excluded','clip_only_excluded','both_retained']}
    rows=[]
    for rec in result['records']:
        gid=rec['group_id']; h=(included|excluded)[gid]; c=old[gid]
        for key in ['original_track_id','mask_id','frame','mask_sha256']:
            if not rec[key]==h[key]==c[key]: raise ValueError('Identity mismatch: '+gid+'/'+key)
        if rec['RGB_sha256']!=c['RGB_sha256']: raise ValueError('RGB mismatch: '+gid)
        human_excluded=gid in excluded
        comp=('both_excluded' if human_excluded else 'auto_only_excluded') if rec['exclude'] else ('human_only_excluded' if human_excluded else 'both_retained')
        cc=('both_excluded' if c['exclude'] else 'semantic_only_excluded') if rec['exclude'] else ('clip_only_excluded' if c['exclude'] else 'both_retained')
        groups[comp].append(gid); versus_clip[cc].append(gid)
        rows.append({**rec,'human_decision':'excluded' if human_excluded else 'retained','comparison':comp,
                     'clip_comparison':cc,'clip_exclude':c['exclude'],'clip_margin':c['margin'],'clip_top_class':c['top_term']})
    audit={}
    for name,ids in [('prior_visual_structure_examples',STRUCTURE_AUDIT),('prior_visual_rug_examples',RUG_AUDIT),('prior_visual_protected_examples',PROTECTED_AUDIT)]:
        audit[name]={'ids':ids,'semantic_excluded':[g for g in ids if automatic[g]['exclude']],
                     'clip_excluded':[g for g in ids if old[g]['exclude']]}
    sensitivity=[]
    for pc in result['config']['sensitivity_pixel_thresholds']:
        for mf in result['config']['sensitivity_mask_thresholds']:
            deleted=[r['group_id'] for r in rows if r['sensitivity_high_excluded_fraction'][str(pc)]>=mf]
            sensitivity.append({'pixel_confidence':pc,'mask_fraction':mf,'excluded':len(deleted),'group_ids':deleted,
                                'manual_retained_excluded':[g for g in deleted if g in included]})
    return {'schema':'auto-v1-seed-semantic-comparison-v1','status':'PASS','counts':{k:len(v) for k,v in groups.items()},
            'groups':groups,'versus_clip':versus_clip,'records':rows,'stats':result['stats'],'clip_stats':clip['stats'],
            'config':result['config'],'model':result['model'],'palette':result['palette'],'frames':result['frames'],
            'human_revision':human['revision'],'manual_reason_labels_available':False,'audit':audit,'sensitivity':sensitivity,
            'audit_note':'Prior assistant visual audit, not user reason labels or an exhaustive independent semantic benchmark.',
            'scope_note':'Human exclusions include duplicate instances. Disagreement is not semantic error. G016 rug is newly allowed for exclusion. No E1 run.'}


def main(args):
    root=Path(args.results); classification=root/'classification.json'
    complete=json.loads((root/'filter_complete.json').read_text()); assert complete['status']=='PASS'
    assert sha(classification)==complete['classification_sha256']
    result=json.loads(classification.read_text()); human=json.loads(Path(args.human_selection).read_text()); clip=json.loads(Path(args.clip_results).read_text())
    data=compare(result,human,clip)
    data.update(classification_sha256=sha(classification),human_reference_sha256=sha(args.human_selection),clip_reference_sha256=sha(args.clip_results))
    save(root/'comparison.json',data)
    shutil.copyfile(args.human_selection,root/'human_reference.json')
    save(root/'suggested_exclusions.json',{'source_manifest_sha256':result['source_manifest_sha256'],
         'classification_sha256':sha(classification),'status':'suggestions_only','groups':[
         {k:r[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256','top_class','high_excluded_fraction']} for r in result['records'] if r['exclude']]})
    for name in ['index.html','review.js','review.css']:
        shutil.copyfile(Path(__file__).parent/name,root/name)
    stats=result['stats']; counts=data['counts']
    lines=['SegFormer whole-frame semantic screening / room2 / 2026-10-18',
           'Model: '+result['config']['model_id']+' @ '+result['config']['model_revision'],
           'Rule fixed before comparison: top-class softmax >= 0.7 and excluded high-confidence pixels / original mask area >= 0.9.',
           'Excluded classes: wall, floor, ceiling, rug. Uncertain masks are retained. No deduplication or E1 run.',
           'Statistics: '+json.dumps(stats), 'Human revision: '+str(human['revision']),
           'Manual comparison: '+json.dumps(counts),
           'Suggested exclusions: '+', '.join(r['group_id'] for r in result['records'] if r['exclude']),
           'Manual retained but suggested excluded: '+', '.join(data['groups']['auto_only_excluded']),
           'Additional exclusions versus CLIP: '+', '.join(data['versus_clip']['semantic_only_excluded']),
           'CLIP excluded but semantic kept: '+', '.join(data['versus_clip']['clip_only_excluded']),
           'Prior assistant visual examples (not an independent benchmark): '+json.dumps(data['audit']),
           'Human exclusions mix semantic categories and duplicates; cannot treat all disagreements as semantic false negatives.',
           'Model softmax probabilities are uncalibrated confidence scores, not empirical correctness probabilities.',
           'Exact masks, original group IDs, RGB sources, official postprocessing agreement and input hashes were checked.']
    (root/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=['comparison.json','classification.json','filter_complete.json','input_freeze.json','seed_manifest.json','seed_id_map.json',
           'suggested_exclusions.json','human_reference.json','report.txt','index.html','review.js','review.css']
    with zipfile.ZipFile(root/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(root/name,name)
        for p in sorted((root/'previews').glob('*.jpg')): z.write(p,p.relative_to(root).as_posix())
    save(root/'delivery_receipt.json',{'sha256':sha(root/'review_bundle.zip'),'bytes':(root/'review_bundle.zip').stat().st_size,
                                    'counts':counts,'stats':stats,'classification_sha256':sha(classification)})
    print(json.dumps({'counts':counts,'groups':data['groups'],'versus_clip':data['versus_clip'],'stats':stats,'audit':data['audit']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--human-selection',required=True);p.add_argument('--clip-results',required=True)
    main(p.parse_args())
