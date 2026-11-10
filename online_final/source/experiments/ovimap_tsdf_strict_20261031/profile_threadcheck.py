from pathlib import Path
import sys,json,shutil,time,cProfile,pstats
C=Path('/home/chenkejun/CVPR/experiments/ovimap_tsdf_strict_20261031');sys.path.insert(0,str(C))
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031');src=O/'ours/room0_400_main';dst=O/'ours/room0_400_threadcheck'
assert not dst.exists();dst.mkdir()
for name in ['history','checkpoints']: (dst/name).mkdir()
commit=json.loads((src/'checkpoint.json').read_text());assert commit['processed_frames']==20
for name in list(commit['history_sha256'])+[commit['association'],commit['votes']]: (dst/name).symlink_to(src/name)
(dst/'checkpoint.json').write_text(json.dumps(commit))
rows=[json.loads(x) for x in (src/'audit.jsonl').read_text().splitlines()]
(dst/'audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['processed_frames']<=20))
import baseline_parallel as worker
sys.argv=['baseline_parallel.py','--scene','room0','--count','400','--native-root',str(O/'native/room0_400_trace'),'--tag','threadcheck','--resume','--stop-after','40']
p=cProfile.Profile();p.enable();worker.main();p.disable()
with (O/'profile_threadcheck.txt').open('w') as f:pstats.Stats(p,stream=f).sort_stats('cumulative').print_stats(30)
