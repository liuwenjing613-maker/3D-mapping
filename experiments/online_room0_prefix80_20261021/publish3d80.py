from pathlib import Path
import json,hashlib,time
C=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix80_20261021')
R=Path('/data/chenkejun/CVPR/results/online_room0_prefix80_20261021')
O=R/'all_edge_20261021'
read=lambda p:json.loads(Path(p).read_text())
def atomic(p,s):
 tmp=p.with_name(p.name+'.prefix80.tmp');tmp.write_text(s);tmp.replace(p)
def link(p,s):
 if p.is_symlink() or p.exists():assert p.resolve()==s.resolve()
 else:p.symlink_to(s.resolve(),target_is_directory=s.is_dir())
W3=Path('/data/chenkejun/CVPR/results/online_room0_prefix50_20261021/review/3d')
new=read(R/'review/3d/manifest.json');assert len(new['maps'])==1 and new['maps'][0]['frames']==80
m=read(W3/'manifest.json');m['maps']=[r for r in m['maps'] if r['frames']!=80]+new['maps']
assert [r['frames'] for r in m['maps']]==[20,50,80]
for p in (R/'review/3d/data').iterdir():link(W3/'data'/p.name,p)
atomic(W3/'manifest.json',json.dumps(m,ensure_ascii=False,indent=2))
C3=Path('/home/chenkejun/CVPR/experiments/online_room0_prefix_compare_20261021')
s=(C/'3d/index_before80.html').read_text().replace('20 / 50 帧','20 / 50 / 80 帧').replace('20 帧 / 50 帧','20 帧 / 50 帧 / 80 帧').replace('50 帧仅含 0–245。','50 帧仅含 0–245；80 帧仅含 0–395。').replace('两个进度','三个进度').replace('href="../">返回 50 帧 ROI 与 mask','href="../frame-edge/?prefix=80">返回留存 mask').replace('app.js?v=1','app.js?v=2')
atomic(C3/'index.html',s)
s=(C3/'app.js').read_text().replace('manifest.maps[1].bbox','manifest.maps[manifest.maps.length-1].bbox')
atomic(C3/'app.js',s)
link(W3/'verification80.json',R/'ply/full_rgb_audit.json')

print('PLY80_PAGE_PUBLISHED')
