from common import *
from appdata import scene_data,metadata,all_stage_rows,card,mask_array,crop_contacts,rows_page,detail,image_bytes
from score_logic import pair_rows,pair_page
import time,sys
started=time.time();locks={};report=[];total=0;grouped=0;rows_checked=0
for scene in SCENES:
 d=scene_data(scene);locks.update(d['locks']);stats=d['stats'];raw=d['raw'];print('AUDIT',scene,len(raw),flush=True)
 assert sum(r=='source' for r in d['excluded'].values())==stats['excluded_cf_masks']
 assert sum(r=='boundary' for r in d['excluded'].values())==stats['excluded_crop_masks']
 assert sum(r=='noise' for r in d['excluded'].values())==stats['excluded_noise_masks']
 for mid,r in raw.items():
  a=mask_array(d,mid)
  if d['excluded'][mid]!='source':crop_contacts(d,mid)
  if 'noise_features' in r:
   f=r['noise_features'];area=f['area'];extent=f['extent'];bbox=f['bbox_area'];core=f['core_pixels']
   removed=area<100 or extent>=80 and 20*core['3']<=area or extent>=80 and 10*area<=bbox and 20*core['8']<=area or extent>=200 and 20*area<=bbox and 20*core['12']<=area
   assert bool(removed)==r['noise_decision']['exclude'],(scene,mid)
  total+=1
 for phase in NAMES:
  rows=all_stage_rows(d,phase);ids=[r['mid'] for r in rows];assert len(ids)==len(set(ids)),(scene,phase)
  if phase in [s['stage'] for s in d['selection']['stages']]:
   st=next(s for s in d['selection']['stages'] if s['stage']==phase)
   assert len(rows)==st['input'],(scene,phase,'input')
   assert sum(r['decision']=='retain' for r in rows)==st['retained'],(scene,phase,'retained')
  for r in rows:
   c=card(d,r,phase);assert c['scores'];rows_checked+=1
  first=rows_page(scene,phase);seen=[]
  for page in range(first['pages']):seen += [r['id'] for r in rows_page(scene,phase,page)['rows']]
  assert seen==ids,(scene,phase,'pagination')
 for gid,g in d['groups'].items():
  rows=pair_rows(scene,'grouping',gid);assert len(rows)==len(g['members'])*(len(g['members'])-1)//2
  page=pair_page(scene,'grouping',gid);allrows=[]
  for p in range(page['pages']):allrows+=pair_page(scene,'grouping',gid,p)['rows']
  assert allrows==rows
  for r in rows:
   if g['primary'] in [r['a'],r['b']]:assert r['passes']
  grouped+=len(g['members'])
 assert set(d['selection']['retained_group_ids'])=={g['id'] for g in read(d['root']/'v2/seed_manifest.json')['groups']}
 # Check representative image encoding and complete per-mask history for excluded and retained masks.
 for mid in list(raw)[:2]+[d['groups'][g]['primary'] for g in d['selection']['retained_group_ids'][:2]]:
  assert image_bytes(scene,mid)[:2]==b'\xff\xd8';assert detail(scene,mid,'retained')['history']
 report.append(dict(scene=scene,raw_masks=len(raw),grouped_masks=len(d['membership']),groups=len(d['groups']),retained=len(d['selection']['retained_group_ids']),stages=[dict(stage=p,input=len(all_stage_rows(d,p)),retained=sum(r['decision']=='retain' for r in all_stage_rows(d,p))) for p in NAMES]))
for p,h in locks.items():assert sha(p)==h
assert sum(r['groups'] for r in report)==637 and sum(r['retained'] for r in report)==196 and grouped==2637
OUT.mkdir(parents=True,exist_ok=True)
result=dict(status='PASS',version='auto-v3-swap / E1 partial',source=str(BASE),counts=report,all_masks_verified=total,grouped_masks=grouped,stage_records_verified=rows_checked,source_hashes=locks,tracking_run=False,repair_run=False,selection_rerun=False,seconds=time.time()-started)
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','counts']},ensure_ascii=False),flush=True)
