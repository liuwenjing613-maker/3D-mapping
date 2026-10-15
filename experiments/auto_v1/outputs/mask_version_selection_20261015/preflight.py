from pathlib import Path
import json
root=Path('/data/chenkejun/CVPR/results')
for p in [root/'overmerge_id_ratio_top20_20261015/web/data.json',root/'e1_roi_top20_20261015/review/viewer_data.json']:
 d=json.loads(p.read_text());print('FILE',p,'KEYS',list(d))
 if 'sources' in d:
  print('SOURCES_TYPE',type(d['sources']).__name__)
  for k,v in list(enumerate(d['sources']))[:3]:print('SOURCE',k,str(v)[:1600])
  print('JOB',str(d['jobs'][0])[:1800]);print('CANDIDATE',str(d['candidates'][0])[:1800])
 else:
  print('CASE',str(d['cases'][0])[:1600]);print('CROP',str(next(iter(d['crops'].values())))[:3200])
for p in [Path('/home/chenkejun/CVPR/AGENTS.md'),Path('/home/chenkejun/CVPR/experiments/AGENTS.md')]:
 if p.exists():print('INSTRUCTIONS',p,p.read_text())
