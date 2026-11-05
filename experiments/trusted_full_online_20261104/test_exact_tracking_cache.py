"""Verify exact archived SAM cache reuse, rejection, ID remapping and prefix fence without inference."""
from pathlib import Path
import ast,json,hashlib,copy,shutil,time
import numpy as np
from PIL import Image
CROOT=Path('/home/chenkejun/CVPR/experiments/trusted_full_online_20261104')
OUT=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104/tests/exact_tracking_cache')
OUT.mkdir(parents=True,exist_ok=True)
SCENE='room0';CHECKPOINT=100;RAW_COUNT=496;GROUP='B'
ROOT=OUT/'root';R=OUT/'tracking';R.mkdir(exist_ok=True)
C=CROOT/'online'/SCENE/'tracking_code'/str(CHECKPOINT)/GROUP
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d))
tree=ast.parse((CROOT/'online'/SCENE/'track_queue.py').read_text())
selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['prior_case','reuse']],type_ignores=[])
exec(compile(selected,'exact_cache_functions','exec'))
oldroot=Path('/data/chenkejun/CVPR/results/unified_full_depth_20261101/online')/SCENE/str(CHECKPOINT)/GROUP/'tracking'
available=read(oldroot/'seed_diagnostics.json')['cases']
base=next((c for c in available if len(c['objects'])>=2),available[0])
case=copy.deepcopy(base)
prior=prior_case(case);assert prior and prior[0]==oldroot
for field,value in [('raw_rgb_sequence_sha256','different_rgb'),('raw_frames',RAW_COUNT+5),('tracking_code_sha256','different_code'),('seed_frame',100000)]:
 bad=copy.deepcopy(case);bad['fingerprint_data'][field]=value
 bad['fingerprint']=hashlib.sha256(json.dumps(bad['fingerprint_data'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert prior_case(bad) is None,field
# Labels are remapped by rank, preserving exact mask geometry and smaller-ID tie order.
oldids=[x['track_id'] for x in case['objects']];newids=[x+1000 for x in oldids]
for obj,new in zip(case['objects'],newids):obj['track_id']=new
uid='verified-remap';case['case_uid']=uid
dest=R/'cases'/uid;dest.mkdir(parents=True,exist_ok=True)
source=np.asarray(Image.open(oldroot/'cases'/base['case_uid']/'prepared_seed.png'),np.uint16)
remapped=np.zeros_like(source)
for a,b in zip(oldids,newids):remapped[source==a]=b
Image.fromarray(remapped).save(dest/'prepared_seed.png')
cfg=read(base['config_path']);cfg['output_dir']=str(dest);cfg['case_uid']=uid;cfg['objects']=[{'track_id':i} for i in newids]
config=dest/'config.json';dump(config,cfg);case['config_path']=str(config)
result=reuse(case,prior)
report=read(dest/'tracking/complete.json');original=read(prior[2])
assert report['track_ids']==newids and report['code_sha256']==digest(C/'track_case.py')
assert len(report['frames'])==RAW_COUNT
for k in [0,cfg['seed_frame'],RAW_COUNT-1]:
 a=np.asarray(Image.open(original['frames'][k]['label_file']),np.uint16)
 b=np.asarray(Image.open(report['frames'][k]['label_file']),np.uint16)
 expected=np.zeros_like(a)
 for x,y in zip(oldids,newids):expected[a==x]=y
 np.testing.assert_array_equal(b,expected)
 for x,y in zip(oldids,newids):assert report['frames'][k]['areas'][str(y)]==original['frames'][k]['areas'][str(x)]
# A separate exact-ID case exercises byte-preserving copyfile (not a hardlink).
case2=copy.deepcopy(base);case2['case_uid']='verified-copy'
dest2=R/'cases'/case2['case_uid'];dest2.mkdir(parents=True,exist_ok=True)
shutil.copyfile(oldroot/'cases'/base['case_uid']/'prepared_seed.png',dest2/'prepared_seed.png')
cfg2=read(base['config_path']);cfg2['output_dir']=str(dest2);cfg2['case_uid']=case2['case_uid']
config2=dest2/'config.json';dump(config2,cfg2);case2['config_path']=str(config2)
reuse(case2,prior)
copied=read(dest2/'tracking/complete.json')
for row,src in zip(copied['frames'],original['frames']):
 assert digest(row['label_file'])==src['label_sha256']
 assert Path(row['label_file']).stat().st_ino!=Path(src['label_file']).stat().st_ino
audit=dict(status='PASS',scene=SCENE,checkpoint=CHECKPOINT,raw_frames=RAW_COUNT,source_case=base['case_uid'],objects=len(oldids),exact_match=True,rejected=['different_rgb','different_prefix','different_tracking_code','different_seed_frame'],remap_preserves_order_and_geometry=True,copy_all_frames_byte_identical=True,no_hardlinks=True,source_report_sha256=digest(prior[2]),no_GPU_inference=True)
dump(OUT/'complete.json',audit);print(json.dumps(audit))
