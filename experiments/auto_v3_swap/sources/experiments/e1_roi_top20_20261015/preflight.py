from pathlib import Path
import collections,json,subprocess
r=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/catalog/room2')
m=json.loads((r/'candidate_manifest.json').read_text()); e=[x for x in m['entries'] if x['rank']<=20]
print(json.dumps({'entries':len(e),'groups':dict(collections.Counter(x['group'] for x in e)),'unique':len({x['uid'] for x in e}),'views':sum(len(x['views']) for x in e),'frames':len({v['frame'] for x in e for v in x['views']})}))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True))
for p in ['/home/chenkejun/CVPR/AGENTS.md','/home/chenkejun/CVPR/experiments/AGENTS.md','/data/chenkejun/CVPR/AGENTS.md','/data/chenkejun/CVPR/results/AGENTS.md']:
 if Path(p).exists():print('INSTRUCTIONS',p,Path(p).read_text())
print('NEW_RESULT_EXISTS',Path('/data/chenkejun/CVPR/results/e1_roi_top20_20261015').exists())
