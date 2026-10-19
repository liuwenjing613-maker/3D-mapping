"""User-requested IoU-only comparison. Original strict run remains immutable."""
from pathlib import Path
import copy,json,shutil,zipfile
from rules import fixed_primary_groups,order_key
from run import sha


def main():
    code=Path(__file__).resolve().parent
    base=Path('/data/chenkejun/CVPR/results/seed_covisible_dedup_20261018')
    out=Path('/data/chenkejun/CVPR/results/seed_covisible_dedup_iou_only_20261018')
    out.mkdir(exist_ok=True,parents=True)
    unchanged_names=['pair_results.json','selection_binary.json','seed_manifest.json']
    previous_hashes={name:sha(out/name) for name in unchanged_names if (out/name).exists()}
    verification=json.loads((base/'verification.json').read_text())
    for name,digest in verification['output_sha256'].items():assert sha(base/name)==digest
    review=json.loads((base/'review_data.json').read_text())
    assert review['input_pair_results_sha256']==sha(base/'pair_results.json')
    original_pairs=json.loads((base/'pair_results.json').read_text())
    manifest_path=Path(review['config']['input_directory'])/'seed_manifest.json'
    assert sha(manifest_path)==review['config']['input_manifest_sha256']
    manifest=json.loads(manifest_path.read_text());bygroup={g['id']:g for g in manifest['groups']}
    ordered=sorted(bygroup,key=lambda g:order_key(bygroup[g]['primary']))
    pair_index={tuple(sorted((p['a'],p['b']))):p for p in original_pairs['pairs']}
    def iou_only(p):return p['union']>0 and 5*p['intersection']>=4*p['union']
    def keep20(p):return iou_only(p) and p['visible_a']>=20 and p['visible_b']>=20
    groups=fixed_primary_groups(ordered,lambda a,b:iou_only(pair_index[tuple(sorted((a,b)))]))
    groups20=fixed_primary_groups(ordered,lambda a,b:keep20(pair_index[tuple(sorted((a,b)))]))
    assert groups==groups20 and all(iou_only(p)==keep20(p) for p in original_pairs['pairs'])
    representative={g:members[0] for members in groups for g in members}
    # Decisions depend exclusively on frozen pair metrics; human labels are used below for comparison only.
    records=copy.deepcopy(review['records'])
    for r in records:
        r['strict_decision']=r['decision'];r['strict_primary_group_id']=r['primary_group_id']
        r['primary_group_id']=representative[r['group_id']]
        r['decision']=r['dedup_decision']='retain' if r['group_id']==r['primary_group_id'] else 'exclude'
    retained=[r for r in records if r['decision']=='retain'];excluded=[r for r in records if r['decision']=='exclude']
    assert len(retained)==56 and len(excluded)==16
    assert {r['group_id'] for r in records if r['strict_decision']=='exclude'} <= {r['group_id'] for r in excluded}
    counts={k:0 for k in ['both_keep','both_exclude','automatic_only_exclude','manual_only_exclude']}
    for r in records:
        h=r['human_decision'];a=r['decision']
        key=('both_keep' if h=='retain' else 'manual_only_exclude') if a=='retain' else ('both_exclude' if h=='exclude' else 'automatic_only_exclude')
        counts[key]+=1
    pairs=copy.deepcopy(review['pairs'])
    for p in pairs:
        p['strict_evidence_sufficient']=p['evidence_sufficient'];p['strict_duplicate']=p['duplicate'];p['strict_reason']=p['reason']
        p['duplicate']=iou_only(p);p['evidence_sufficient']=p['union']>0
        p['reason']='duplicate' if p['duplicate'] else 'overlap_below_threshold'
    group_records=[dict(primary=g[0],members=g,support_count=bygroup[g[0]]['primary']['support_count']) for g in groups]
    cfg={**review['config'],'mode':'iou_only','minimum_visible_fraction_each':None,'minimum_visible_vertices_each':None,
         'output_directory':str(out),'comparison_parent':str(base),'decision_rule':'nonempty union and common-visible IoU >= 0.8 only'}
    stats={**review['stats'],'retained':56,'excluded':16,'duplicate_pairs':18,
           'sufficient_evidence_pairs':sum(p['evidence_sufficient'] for p in pairs),
           'reason_counts':{'duplicate':18,'overlap_below_threshold':2538}}
    review.update(config=cfg,stats=stats,pairs=pairs,groups=group_records,records=records)
    review['manual_comparison']['counts']=counts
    comparison=dict(strict={'excluded':2,'retained':70,'passing_pairs':2},iou_only={'excluded':16,'retained':56,'passing_pairs':18},
        keep20_is_identical=True,extra_excluded=[r['group_id'] for r in excluded if r['strict_decision']=='retain'],
        human_retained_now_excluded=[r['group_id'] for r in excluded if r['human_decision']=='retain'],
        manual_counts=counts,groups=[g for g in group_records if len(g['members'])>1])
    def save(name,data):(out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    confirmation_path=code/'human_confirmation.json'
    if confirmation_path.exists():
        confirmation=json.loads(confirmation_path.read_text())
        record_index={r['group_id']:r for r in records}
        adjusted_labels={r['group_id']:r['human_decision'] for r in records}
        for c in confirmation['confirmations']:
            assert c['same_instance'] and set(c['pair'])=={c['retain'],c['exclude']}
            assert representative[c['exclude']]==c['retain'] and representative[c['retain']]==c['retain']
            c['identities']=[{k:record_index[g][k] for k in ['group_id','mask_id','frame','mask_sha256']} for g in c['pair']]
            adjusted_labels[c['retain']]='retain';adjusted_labels[c['exclude']]='exclude'
        adjusted_counts={k:0 for k in counts}
        for r in records:
            h=adjusted_labels[r['group_id']];a=r['decision']
            key=('both_keep' if h=='retain' else 'manual_only_exclude') if a=='retain' else ('both_exclude' if h=='exclude' else 'automatic_only_exclude')
            adjusted_counts[key]+=1
        review['user_confirmations']=confirmation['confirmations']
        review['manual_comparison']['after_latest_confirmation']=adjusted_counts
        comparison['user_confirmations']=confirmation['confirmations']
        comparison['after_latest_confirmation']=adjusted_counts
        save('human_confirmation.json',confirmation)
    # Compact selection records include identity and explicit decisions but no human labels.
    keys=['group_id','original_track_id','mask_id','frame','mask_sha256','semantic_decision','decision','dedup_decision','primary_group_id','support_count']
    save('selection_binary.json',dict(schema='auto-v1-covisible-iou-only-selection-v1',counts=dict(retained=56,excluded=16),
         input_selection_sha256=cfg['input_selection_sha256'],groups=group_records,
         retained=[{k:r[k] for k in keys} for r in retained],excluded=[{k:r[k] for k in keys} for r in excluded]))
    result_manifest=copy.deepcopy(manifest)
    result_manifest.update(groups=[bygroup[r['group_id']] for r in retained],dedup_config=cfg,dedup_groups=group_records,
         dedup_input_manifest_sha256=cfg['input_manifest_sha256'],dedup_excluded_group_ids=[r['group_id'] for r in excluded],
         status='dedup_iou_only_comparison',tracking_requested=False,map_write=False)
    save('seed_manifest.json',result_manifest)
    rawpairs=[{k:v for k,v in p.items() if k!='images'} for p in pairs]
    save('pair_results.json',dict(config=cfg,stats=stats,pairs=rawpairs,groups=group_records,
        records=[{k:v for k,v in r.items() if k not in ['human_decision','images']} for r in records]))
    for name,digest in previous_hashes.items():assert sha(out/name)==digest,('Confirmation must not alter automatic outputs',name)
    review['input_pair_results_sha256']=sha(out/'pair_results.json')
    save('review_data.json',review);save('comparison.json',comparison)
    if not (out/'masks').exists():(out/'masks').symlink_to((base/'masks').resolve(),target_is_directory=True)
    if not (out/'previews').exists():(out/'previews').symlink_to((base/'previews').resolve(),target_is_directory=True)
    for r in retained:assert sha(out/bygroup[r['group_id']]['primary']['mask'])==r['mask_sha256']
    save('verification.json',dict(status='PASS',base_verification_sha256=sha(base/'verification.json'),
        base_pair_results_sha256=sha(base/'pair_results.json'),code_sha256=sha(code/'ablation.py'),
        checks=['All immutable original pair metrics reused without alteration','Pure IoU and remove fraction but keep minimum 20 points give identical 18 passing pairs and 56 groups',
                'Fixed representative; every deleted seed has a direct passing edge to kept representative','Original 2 deletions are subset of 16',
                '56 unchanged retained masks verified; original IDs preserved','Manual comparison is post-decision only; no E1 run'],
        output_sha256={n:sha(out/n) for n in ['review_data.json','pair_results.json','selection_binary.json','seed_manifest.json','comparison.json','human_confirmation.json'] if (out/n).exists()}))
    for name in ['index.html','review.js','review.css']:shutil.copyfile(code/name,out/name)
    save('complete.json',dict(status='PASS',stats=stats))
    package=['review_data.json','comparison.json','pair_results.json','selection_binary.json','seed_manifest.json','verification.json','complete.json','index.html','review.js','review.css']
    if (out/'human_confirmation.json').exists():package.append('human_confirmation.json')
    with zipfile.ZipFile(out/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in package:z.write(out/name,name)
        for p in sorted((out/'previews').glob('*.jpg')):z.write(p,'previews/'+p.name)
    receipt=dict(sha256=sha(out/'review_bundle.zip'),bytes=(out/'review_bundle.zip').stat().st_size,stats=stats)
    save('delivery_receipt.json',receipt)
    print(json.dumps(comparison));print(json.dumps(receipt))


if __name__=='__main__':main()
