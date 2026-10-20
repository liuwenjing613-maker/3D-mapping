"""Publish the completed 35-mask run with historical 27-mask and manual controls."""
from pathlib import Path
import copy,hashlib,importlib.util,json,shutil,time
import numpy as np
from PIL import Image
C=Path(__file__).resolve().parent
R=Path('/data/chenkejun/CVPR/results/auto_v2_room0_pr35_20261019')
OLD=Path('/data/chenkejun/CVPR/results/auto_v2_room0_20261019')
OC=Path('/home/chenkejun/CVPR/experiments/auto_v2_room0_20261019')
O=R/'review';A=O/'assets'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert read(R/'complete.json')['status']=='PASS'
 assert read(R/'comparison_four_way/consistency.json')['status']=='PASS'
 A.mkdir(parents=True,exist_ok=True)
 for p in (OLD/'review/assets').iterdir():
  if p.name.endswith('_after.jpg'):continue
  q=A/p.name
  if not q.exists():q.symlink_to(p)
 data=copy.deepcopy(read(OLD/'review/data.json'));selection=read(R/'v2/selection.json')
 dec={r['group_id']:r for r in selection['records']}
 for row in data['rows']:row.update(dec[row['group_id']])
 assert sum(r['decision']=='retain' for r in data['rows'])==35
 import sys
 sys.path.insert(0,str(OC))
 spec=importlib.util.spec_from_file_location('old_review_builder',OC/'build_review.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
 byid={r['group_id']:r for r in data['rows']};masks={gid:np.asarray(Image.open(A/(gid+'_mask.png')))>0 for gid in byid}
 source_frames={f['frame']:f for f in read(R/'v2/semantic/classification.json')['frames']}
 for frame in data['frames']:
  retained=[byid[g] for g in frame['group_ids'] if byid[g]['decision']=='retain']
  original_rgb=Path(source_frames[frame['frame']]['RGB_path']);assert sha(original_rgb)==source_frames[frame['frame']]['RGB_sha256']
  rgb=np.asarray(Image.open(original_rgb).convert('RGB'))
  Image.fromarray(builder.collection_image(rgb,retained,masks)).save(O/frame['after_image'],quality=90)
  frame['after']=len(retained)
 data.pop('three_way',None)
 data.update(generated=time.time(),selection=selection,evaluation=read(R/'e1_repair/evaluation_r2/comparison.json'),four_way=read(R/'comparison_four_way/summary.json'),code_root=str(C),result_root=str(R),source_hashes={n:sha(R/n) for n in ['v1/seed_manifest.json','v1/audit.json','v2/selection.json','v2/measurements.json','v2/seed_manifest.json','v2/source_lock.json']})
 data['paired_evaluation']={s:read(R/'e1_repair/evaluation_r2/paired'/s/'paired_repair_metrics.json') for s in ['native','final']}
 (O/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
 for p in (C/'review_web').iterdir():shutil.copy2(p,O/p.name)
 reports=O/'reports';reports.mkdir(exist_ok=True)
 for p in (OLD/'review/reports').iterdir():
  if p.name=='README.md' or p.name.startswith(('comparison_three_way','e1_repair','v2__')):continue
  q=reports/p.name
  if not q.exists():q.symlink_to(p)
 (reports/'README.md').write_text('room0 自动 35 张 + 原版 E1\n\n使用原 auto-v1 的 103 张种子；第 1–4 步不变，只对 SegFormer 主类别为 windowpane/cabinet 的第 5 步输入豁免 p/r 排除。新增 G026/G027/G058/G063/G069/G097/G102/G103，原 27 张全部保留。M0466 保留 CropFormer 来源纠正。\n\n18 组完整追踪经输入与全部输出哈希核验后复用，6 组重新推理；原版 E1 全阶段重新计算。四组在 main 默认 R2 下重新评分，验证共同几何、RGB、400 帧、72 GT、observed support、固定对应索引和配置；旧三组指标须精确复现。\n\n两份 PLY 是最终 TSDF 的互斥完整分区：正实例彩色点；未分配点按 U=黑、T=黄、C=红显示。没有降采样。')
 for group,names in [('v2',['selection.json','measurements.json','seed_manifest.json','source_lock.json','complete.json']),('comparison_four_way',['summary.json','consistency.json','inputs_frozen_before_scoring.json']),('e1_repair',['run_complete.json','selected_input_audit.json','pipeline_code_freeze.json','pipeline_complete.json','evaluation_r2/comparison.json','evaluation_r2/protocol_freeze.json','evaluation_r2/per_GT.json','evaluation_r2/paired/native/paired_repair_metrics.json','evaluation_r2/paired/final/paired_repair_metrics.json'])]:
  for name in names:shutil.copy2(R/group/name,reports/(group+'__'+name.replace('/','__')))
 ply=O/'ply';ply.mkdir(exist_ok=True)
 manifest=read(R/'ply_black_yellow_red_20261019/manifest.json')
 for row in manifest['files']:
  p=R/'ply_black_yellow_red_20261019'/row['file'];assert sha(p)==row['sha256']
  q=ply/p.name
  if not q.exists():q.symlink_to(p)
 shutil.copy2(R/'ply_black_yellow_red_20261019/manifest.json',ply/'manifest.json')
 files={p.relative_to(O).as_posix():sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='review_manifest.json'}
 result=dict(status='PASS',files=files,evaluation_complete=True,seeds=103,retained=35,four_way=True,time=time.time())
 (O/'review_manifest.json').write_text(json.dumps(result,indent=2));print('REVIEW35_COMPLETE',len(files))

if __name__=='__main__':main()
