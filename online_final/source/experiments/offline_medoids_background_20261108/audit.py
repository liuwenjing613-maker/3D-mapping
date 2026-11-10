"""Independent pixel-count and input-selection validation for semantic replay."""
import sys,json,collections
from pathlib import Path
import numpy as np
from PIL import Image
from replay import INPUT,OUT,VERSION,sha
sys.path.insert(0,'/home/chenkejun/CVPR/experiments/offline_mask_filters_196_20261023')
from appdata import scene_data,mask_array
from common import SCENES

checks=[]
for scene in SCENES:
 result=json.loads((OUT/scene/'classification.json').read_text())
 medoids=json.loads((INPUT/(scene+'.json')).read_text());data=scene_data(scene)
 expected={(gid,c['number'],c['representative']) for gid,g in medoids['groups'].items() for c in g['clusters']}
 observed={(r['group_id'],r['cluster'],r['mask_id']) for r in result['records']}
 assert expected==observed and len(observed)==len(result['records'])
 assert result['version']==VERSION and result['GT_used'] is False
 cfg=result['config'];assert cfg['pixel_confidence']==.7 and cfg['mask_fraction']==.9
 labels={int(k):v for k,v in result['model']['id2label'].items()}
 excluded={k for k,v in labels.items() if v in ('wall','floor','ceiling','rug')};assert excluded=={0,3,5,28}
 by_frame=collections.defaultdict(list)
 for r in result['records']:by_frame[r['frame']].append(r)
 assert set(by_frame)=={f['frame'] for f in result['frames']}
 for f in result['frames']:
  assert sha(f['cache'])==f['cache_sha256'] and sha(f['RGB_path'])==f['RGB_sha256']
  with np.load(f['cache'],allow_pickle=False) as z:prediction=z['labels'];confidence=z['confidence']
  for r in by_frame[f['frame']]:
   mid=r['mask_id'];mask=mask_array(data,mid);ids=prediction[mask];conf=confidence[mask];area=int(np.count_nonzero(mask))
   assert r['area']==area and r['mask_sha256']==data['raw'][mid]['mask_sha256']
   bg=np.zeros(len(ids),bool)
   for k in excluded:bg|=(ids==k)
   high=conf>=.7;nb=int(np.count_nonzero(bg&high));nf=int(np.count_nonzero(~bg&high));nu=int(np.count_nonzero(~high))
   assert nb+nf+nu==area
   assert (r['high_excluded_pixels'],r['high_retained_pixels'],r['low_confidence_pixels'])==(nb,nf,nu)
   assert r['exclude']==(10*nb>=9*area)
   decision='suggest_exclude' if 10*nb>=9*area else 'retain' if 10*nf>=9*area else 'review'
   assert r['decision']==decision and r['operational_decision']==('exclude' if r['exclude'] else 'retain')
   for key,num in [('high_excluded_fraction',nb),('high_retained_fraction',nf),('low_confidence_fraction',nu)]:assert r[key]==num/area
   counts=np.bincount(ids,minlength=150);hi=np.bincount(ids[high],minlength=150)
   assert sum(c['pixels'] for c in r['categories'])==area and r['top_class']==labels[int(np.argmax(counts))]
   assert r['top_class_fraction']==int(counts.max())/area
   for c in r['categories']:assert c['pixels']==int(counts[c['id']]) and c['high_confidence_pixels']==int(hi[c['id']])
   for name,p in r['images'].items():
    with Image.open(OUT/scene/p) as im:im.verify()
 kept=json.loads((OUT/scene/'retained_candidates.json').read_text())['candidates']
 assert {r['mask_id'] for r in kept}=={r['mask_id'] for r in result['records'] if not r['exclude']}
 summary=result['summary'];assert summary['input']==len(result['records'])
 assert summary['retained']==len(kept) and summary['excluded']==summary['input']-len(kept)
 assert summary['review_retained']==sum(r['decision']=='review' for r in result['records'])
 checks.append(dict(scene=scene,status='PASS',summary=summary));print(scene,'PASS',summary,flush=True)
audit=json.loads((OUT/'audit.json').read_text())
assert audit['input']==975 and audit['input']==sum(c['summary']['input'] for c in checks)
assert all(sha(p)==h for p,h in audit['source_locks'].items())
assert all(sha(p)==h for p,h in audit['cache_locks'].items())
parity=json.loads((OUT/'inference_parity.json').read_text())
assert parity['labels_equal'] and parity['confidence_gate_equal']
audit.update(status='PASS',independent_validation=checks,inference_parity=parity,validation='Every selected representative covered exactly once; independent integer pixel counts and 90% boundary; uncertainty retained; preview files verified; source and cached predictions unchanged.')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
print('AUDIT_PASS',audit['input'],audit['retained'],audit['excluded'])
