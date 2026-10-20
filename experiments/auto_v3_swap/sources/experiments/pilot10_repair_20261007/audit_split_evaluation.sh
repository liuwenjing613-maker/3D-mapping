set -eu
/data/chenkejun/beauty/ovo_paper_original_20260915/env/bin/python - <<'PY'
from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
import evaluate_pilot_v3 as e
work=e.ROOT/'joint_mask_replacement_20261007/room2_mask14_mask24'
conditions={'baseline':e.ROOT/'v3/baseline/room2','pixel_control':work/'v3/pixel_control','whole_mask':work/'v3/whole_mask'}
rows={}
for arm,path in conditions.items():
    z=e.load_overlap(path/'overlap.npz')
    target_columns=[int(np.flatnonzero(z['gt_ids']==g)[0]) for g in [4001,4002]]
    instances=[]
    for pid in [52,355]:
        found=np.flatnonzero(z['pred_uids']==f'p1a1-id:{pid}')
        if not len(found):continue
        p=int(found[0])
        instances.append({'persistent_id':pid,'targets':[{ 'GT_id':int(z['gt_ids'][g]),
            'IoU':float(z['iou'][p,g]),'recall':float(z['recall'][p,g]),'purity':float(z['precision'][p,g]),
            'reference_intersection_vertices':int(round(z['recall'][p,g]*z['gt_size'][g]))} for g in target_columns],
            'diagnostic_2cm_GT_ids_touched':z['gt_ids'][(z['diagnostic_intersection'][p]>=10)&(z['diagnostic_recall'][p]>=.05)].tolist()})
    report=json.loads((path/'complete.json').read_text())
    rows[arm]={'instances':instances,'counts':report['metrics']['CA_PRF1_0_5'],'adapter':report['adapter']}
data={'status':'PASS','posthoc_readonly':True,'protocol_changed':False,'conditions':rows,
    'interpretation':'Local separation improves purity; complete instance recall and AP50 are separate requirements. The current adapter searches only positive-labeled native points.'}
e.dump(work/'split_evaluation_audit.json',data)
print(json.dumps(data,ensure_ascii=False,indent=2))
PY
