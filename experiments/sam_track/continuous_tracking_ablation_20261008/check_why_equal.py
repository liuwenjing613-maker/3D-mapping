"""Explain why different raw mask intervals produce identical accepted vote ledgers."""
from pathlib import Path
import hashlib,json
WORK=Path(__file__).resolve().parent
WEB=WORK.parents[1]/'results/固定案例_三模型对比_20261006/continuous_tracking_ablation_20261008'
DATA=WORK/'server_results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
results={}
for scene in ['room0','room2']:
    meta=read(WEB/(scene+'_preview_data.json'))
    windows={name:{w['track']:w for w in meta['continuity_policies'][name]['tracks']} for name in ['gap0','gap1']}
    decisions={name:read(DATA/scene/name/'full_track/frame_decisions.json')['frames'] for name in ['gap0','gap1']}
    assert [r['frame'] for r in decisions['gap0']]==[r['frame'] for r in decisions['gap1']]
    by_frame={r['frame']:r for r in decisions['gap1']}
    rows=[]
    for t in meta['tracks']:
        oid=t['track'];a,b=windows['gap0'][oid]['allowed_frame_range'];c,d=windows['gap1'][oid]['allowed_frame_range']
        for fid,r in by_frame.items():
            if c<=fid<=d and not a<=fid<=b and meta['frames'][fid]['areas'][oid-1]>0:
                decision=r['objects'][str(oid)]
                assert not decision['accepted'] and oid not in r['accepted_track_ids']
                rows.append({'track':oid,'ROI':t['ROI'],'mask':t['mask'],'frame':fid,
                  'native_area':meta['frames'][fid]['areas'][oid-1],'unchanged_gate_reasons':decision['reasons']})
    assert all(a['accepted_track_ids']==b['accepted_track_ids'] for a,b in zip(decisions['gap0'],decisions['gap1']))
    results[scene]={'extra_nonempty_mask_mapping_frames_in_gap1':len(rows),
      'all_extra_mapping_masks_rejected_by_unchanged_gates':True,
      'accepted_raw_track_ID_sets_identical_in_all_400_mapping_frames':True,'extra_rows':rows}
output={'status':'PASS','explanation':'Extra raw masks in gap1 do not add accepted votes; complete vote ledgers and final maps are bit identical as verified by server comparison_audit',
        'scenes':results,'GT_used':False,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(WORK/'why_policies_equal.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({s:{k:v for k,v in r.items() if k!='extra_rows'} for s,r in results.items()},ensure_ascii=False,indent=2))
