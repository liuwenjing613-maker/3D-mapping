"""Validate recomputed main candidates on native geometry; no online-only border rule."""
from common_continue import *
import numpy as np
from PIL import Image
def main():
 started=time.time();assert CHECKPOINT==400 and OLD_COUNT==0
 mapping=read(B/'mapping_complete.json');assert mapping['status']=='PASS' and mapping['frame_ids']==FIDS
 for path,h in read(CANDIDATE_ROOT/'mapping_input_freeze.json')['source_hashes'].items():assert digest(path)==h,path
 cat=read(CANDIDATE_ROOT/'catalog'/SCENE/'candidate_manifest.json')
 assert cat['candidate_search_frame_ids']==FIDS and cat['scoring_history_frame_ids']==FIDS
 grouping=read(S/'grouping.json');chosen=read(CHOSEN);selection=read(CANDIDATE_ROOT/'v2/selection.json')
 assert set(g['id'] for g in chosen['groups'])==set(selection['retained_group_ids'])
 for row in grouping['seeds']:
  path=S/row['mask'];assert digest(path)==row['mask_sha256']
  mask=np.asarray(Image.open(path))>0;assert mask.shape==(680,1200) and int(mask.sum())==row['area']
  assert row['frame'] in FIDS and not row['noise_decision']['exclude']
 for path,h in grouping['source_sha256'].items():assert digest(path)==h,path
 blocked=[]
 for p in (ROOT/'io_audit').glob('*.jsonl'):
  for line in p.read_text().splitlines():
   row=json.loads(line)
   if row.get('blocked') and not row.get('probe'):blocked.append(row)
 assert not blocked,blocked[:3]
 report=dict(status='PASS',checkpoint=400,group=GROUP,retained=len(chosen['groups']),source_manifest=str(CHOSEN),
  source_manifest_sha256=digest(CHOSEN),candidate_raw_frame_range=[0,1995],GT_used=False,
  source_masks_unchanged=True,current_branch_mapping_unchanged=True,forbidden_nonprobe_accesses=0,
  main_variant='single final partial+core+E1, swap filter order, window/cabinet routing, original crop/noise rules',
  historical_exceptions=read(CANDIDATE_ROOT/'main_exceptions.json'),seconds=time.time()-started)
 dump(CANDIDATE_ROOT/'online_qa.json',report)
if __name__=='__main__':main()
