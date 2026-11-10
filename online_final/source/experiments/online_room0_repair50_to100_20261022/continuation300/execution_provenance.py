"""Execution source snapshots for mapping/commit audit; no numerical behavior changes."""
from pathlib import Path
import hashlib,json,time
C=Path('/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/continuation300')
BASE=C.parent
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
SNAP=Path('/home/chenkejun/CVPR/experiments/p1a1_raw_replica8_20261002/snapshot/revisable_instance_map/src/revisable_instance_map')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_bytes(data);tmp.replace(path)
def dump(path,obj):write(path,(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode())
def source_paths(stage,n,g):
 if stage=='mapping':
  return [C/'map_continue.py',C/'map_support.py',C/'map_guard/sitecustomize.py',BASE/'prefix100/one_vote.py']+sorted(SNAP.glob('*.py'))
 assert stage=='commit'
 frozen=MAIN/'instance_repair/frozen_e1_20261012/room0'
 mapping=json.loads((frozen/'module_map.json').read_text())
 paths=[C/'commit_continue.py',C/'bootstrap_continue.py',C/'common_continue.py',C/'scene_continue.py',C/'selection_guard/sitecustomize.py',MAIN/'instance_repair/cli.py',frozen/'module_map.json']
 paths+=sorted((C/'tracking_code'/str(n)/g).glob('*.py'))
 for value in mapping.values():
  p=Path(value);paths.append(p if p.is_absolute() else frozen/p)
 return sorted(set(paths))
def snapshot(folder,stage,n,g,overrides=None,started_unix=None):
 folder=Path(folder);dest=folder/'execution_code';manifest=dest/'manifest.json'
 assert not manifest.exists(),str(manifest)
 rows=[];overrides=overrides or {}
 for i,path in enumerate(source_paths(stage,n,g)):
  raw=path.read_bytes();executed=overrides.get(str(path),raw)
  local=dest/'files'/(str(i).zfill(3)+'_'+path.name);write(local,executed)
  stat=path.stat();origin='prelaunch_source_snapshot' if started_unix is None else 'postlaunch_source_capture'
  if str(path) in overrides:origin='exact_prechange_text_recovered_from_session'
  rows.append(dict(source=str(path),snapshot=str(local),sha256=sha(executed),current_source_sha256_at_capture=sha(raw),
   source_mtime=stat.st_mtime,source_mtime_precedes_launch=None if started_unix is None else stat.st_mtime<=started_unix,origin=origin))
 report=dict(status='CAPTURED',stage=stage,checkpoint=n,group=g,captured_unix=time.time(),prelaunch=started_unix is None,
  started_unix=started_unix,files=rows,numerical_behavior_changed=False)
 dump(manifest,report)
 return dict(code_manifest=str(manifest),code_manifest_sha256=sha(manifest.read_bytes()))
def verify(folder):
 folder=Path(folder);manifest=folder/'execution_code/manifest.json';obj=json.loads(manifest.read_text());rows=[]
 for row in obj['files']:
  assert sha(Path(row['snapshot']).read_bytes())==row['sha256'],row['snapshot']
  same=sha(Path(row['source']).read_bytes())==row['current_source_sha256_at_capture']
  rows.append(dict(source=row['source'],unchanged_since_capture=same))
  assert same,row['source']
 out=folder/'execution_code/verification.json'
 dump(out,dict(status='PASS',manifest=str(manifest),manifest_sha256=sha(manifest.read_bytes()),files=rows,
  checked_unix=time.time(),scope='Snapshot contents match and source files unchanged since capture; retrospective capture is explicitly marked in manifest.'))
 return str(out)
