"""Assemble requested results and only source bytes absent from the parent."""
from pathlib import Path
import hashlib,json,subprocess,shutil,sys,zipfile,difflib
W=Path(__file__).resolve().parents[2]
O=Path(__file__).parent
REPO=W/'worktrees/scannet-result'
PKG=REPO/'experiments/scannet_result_20261020'
BASE='1103d8196a980874ef743f572d3b4c0748a4fdab'
DELIVERY=W/'results/ScanNet_E1_auto-v3-swap_partial_20261020'
SCENES=sys.argv[1:] or ['scene0011_00']
sha=lambda b:hashlib.sha256(b).hexdigest()
git=lambda *a:subprocess.check_output(['git','-C',str(REPO),*a])
assert git('branch','--show-current').decode().strip()=='scannet-result'
PKG.mkdir(parents=True,exist_ok=True)
(PKG/'.gitattributes').write_text('* -text\ncode/** -whitespace\nevidence/** -whitespace\nresults/** -whitespace\ndiffs/** -whitespace\n',encoding='utf-8',newline='\n')
tracked={}
paths=set()
for item in git('ls-tree','-r','-z',BASE).split(b'\0'):
    if not item:continue
    meta,path=item.split(b'\t',1);mode,kind,blob=meta.split()
    if kind!=b'blob':continue
    path=path.decode();paths.add(path);tracked.setdefault(blob.decode(),[]).append(path)
for blob,items in tracked.items():
    items.sort(key=lambda p:(not p.startswith('experiments/auto_v3_swap/'),len(p),p))
snapshots=[]
source={}
origins={}
for scene in SCENES:
    archive=O/'github_download'/f'evidence_{scene}.zip'
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('snapshot_manifest.json'))
        assert manifest['scenes']==[scene]
        for rel,record in manifest['files'].items():
            assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
            data=z.read(rel);assert sha(data)==record['sha256']
            if rel.startswith('source_snapshot/'):
                name=rel.removeprefix('source_snapshot/')
                if name in source:assert source[name]==data,(name,'source changed between snapshots')
                source[name]=data;origins[name]=record
            else:
                target=PKG/'evidence'/rel.removeprefix('extra_evidence/')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or target.read_bytes()!=data:target.write_bytes(data)
        snapshots.append(dict(scene=scene,archive_sha256=sha(archive.read_bytes()),files=manifest['files']))
provenance=json.loads(source.get('adapter_changes.json',b'{"changes":[]}'))
changes={x['file']:x for x in provenance['changes']}
inventory=[]
for name,data in sorted(source.items()):
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    row=dict(runtime_relative_path=name,sha256=sha(data),bytes=len(data),server_source=origins[name]['source'])
    if blob in tracked:
        inherited=tracked[blob][0]
        assert sha(git('show',BASE+':'+inherited))==row['sha256']
        row.update(disposition='unchanged_inherited',repository_path=inherited,base_commit=BASE)
    else:
        target=PKG/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        row.update(disposition='added_or_changed',repository_path=target.relative_to(REPO).as_posix())
        original=changes.get(name,{}).get('source','').replace('\\','/')
        candidate=None
        if '/experiments/auto_v2/' in original:
            tail=original.split('/experiments/auto_v2/',1)[1]
            family,rest=tail.split('/',1)
            prefix=('auto_v2_'+family if family.startswith('replica6_') else family)
            candidate='experiments/auto_v3_swap/sources/experiments/'+prefix+'/'+rest
        elif '/instance_repair/' in original:
            candidate='instance_repair/'+original.split('/instance_repair/',1)[1]
        if candidate in paths:
            old=git('show',BASE+':'+candidate)
            row.update(comparison_source=candidate,comparison_source_sha256=sha(old))
            if name.endswith(('.py','.json')):
                diff=''.join(difflib.unified_diff(old.decode('utf-8').splitlines(True),
                    data.decode('utf-8').splitlines(True),fromfile=candidate,tofile='code/'+name))
                dest=PKG/'diffs'/(name+'.patch');dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_text(diff,encoding='utf-8',newline='\n')
    inventory.append(row)
(PKG/'source_inventory.json').write_text(json.dumps(dict(base_commit=BASE,records=inventory,
    unchanged=sum(r['disposition']=='unchanged_inherited' for r in inventory),
    added_or_changed=sum(r['disposition']=='added_or_changed' for r in inventory)),ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
(PKG/'evidence_snapshot_manifest.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
dependencies=[]
dep_archive=O/'github_download/external_dependencies.zip'
if dep_archive.exists():
    with zipfile.ZipFile(dep_archive) as z:
        dep_manifest=json.loads(z.read('dependency_manifest.json'))
        for name,record in sorted(dep_manifest['files'].items()):
            data=z.read('external_sources/'+name);assert sha(data)==record['sha256']
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            row=dict(record,external_relative_path=name)
            if blob in tracked:
                inherited=tracked[blob][0]
                row.update(disposition='unchanged_inherited',repository_path=inherited,base_commit=BASE)
            else:
                target=PKG/'code/external'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                row.update(disposition='added_or_changed',repository_path=target.relative_to(REPO).as_posix())
            dependencies.append(row)
    (PKG/'external_dependencies_inventory.json').write_text(json.dumps(dict(records=dependencies,
        archive_sha256=sha(dep_archive.read_bytes()),scope=dep_manifest['scope']),ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
delivery=json.loads((DELIVERY/'delivery_manifest.json').read_text(encoding='utf-8'))
assert delivery['completed_scenes']==SCENES
for rel,digest in delivery['sha256'].items():
    p=DELIVERY/rel;assert sha(p.read_bytes())==digest
    target=PKG/'results'/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
shutil.copy2(DELIVERY/'delivery_manifest.json',PKG/'results/delivery_manifest.json')
for scene in SCENES:
    src=O/'github_analysis'/scene
    assert (src/f'analysis_{scene}.md').exists(),src
    shutil.copytree(src,PKG/'results'/scene,dirs_exist_ok=True)
# Include reproducible adapter generation and approved filtering integration, not SSH credentials.
for name in ['build.py','patch_regroup.py','build_github_publication.py']:
    dest=PKG/'tools'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(O/name,dest)
shutil.copy2(O/'publication_verify.py',PKG/'verify.py')
for p in (O/'github_analysis').glob('analyze_scene*.py'):
    shutil.copy2(p,PKG/'tools'/p.name)
rows=json.loads((DELIVERY/'comparison_all.json').read_text(encoding='utf-8'))
lines=['# ScanNet：严格一票无修复、E1 与 auto-v3-swap partial','',
       '本分支保存两个既定 ScanNet 场景的实测结果、逐 GT 分析和实现差异。唯一修复收益基线是**严格一票无修复**。历史结果单列，仅供参考。',
       '',f"当前已完成并归档：{', '.join(SCENES)}。",'',
       '| 场景 | 方法 | AP% | AP50% | F1% | PQ% | TP/FP/FN |','|---|---|---:|---:|---:|---:|---|']
names={'baseline':'严格一票无修复','E1':'E1 whole','swap_partial':'auto-v3-swap partial'}
for r in rows['rows']:
    if r['stage']=='final':
        lines.append(f"| {r['scene']} | {names[r['method']]} | {100*r['AP']:.4f} | {100*r['AP50']:.4f} | {100*r['F1']:.4f} | {100*r['PQ']:.4f} | {r['TP']}/{r['FP']}/{r['FN']} |")
lines+=['','当前已完成场景中，E1与partial的F1、PQ均低于严格一票无修复基线；partial和E1整体接近。以下计数来自固定IoU 0.5评分，不能只按局部IoU改善个数判断总体收益。','']
for scene in SCENES:
    current={r['method']:r for r in rows['rows'] if r['scene']==scene and r['stage']=='final'}
    b,e=current['baseline'],current['E1']
    lines.append(f"- {scene}：TP {b['TP']}→{e['TP']}，FP {b['FP']}→{e['FP']}；E1相对严格基线F1 {100*(e['F1']-b['F1']):+.4f}、PQ {100*(e['PQ']-b['PQ']):+.4f}个百分点。")
lines+=['','## 分析与结果','']
for scene in SCENES:
    lines.append(f"- [{scene} 分析](results/{scene}/analysis_{scene}.md)：逐 GT、结构诊断、E1/partial 差异及局限。")
    lines.append(f"- [{scene} PLY](results/{scene}/ply)、[原生/最终 NPZ 预测](evidence/{scene}/prediction_maps)、[完整评分](results/{scene}/evaluation_r2/comparison.json)。")
lines+=['','[总指标和历史参考](results/comparison_all.json) · [中文对比说明](results/对比说明.txt) · [实现差异说明](IMPLEMENTATION_DIFFS.md)','',
    '## 固定评价口径','',
    'scene0011_00 为 238 帧 / 27 GT；scene0608_00 为 339 帧 / 34 GT。使用服务器现有步长10序列，保留原始帧号。新三组共用相同 TSDF、关联、GT范围和对应索引。',
    '评分导入 main 工作区锁定的 R2 实现，配合历史已审核的 ScanNet 场景范围适配；不是把 Replica 的350 GT套到ScanNet。评分源码基准 `cd260569e89de1e9525f82c45ee10842b80616b0`，审核归档 `d035f68770cefab27c9751e97062c72faa07367c`，评价 `frozen=false`。main默认CLI当前限定Replica，本次实际入口是归档的 `code/evaluate.py`。',
    '历史 P1-A1 使用反向投影投票；本次严格基线使用 E1 正向投影与同帧同点严格一票，因此跨历史结果的差异不能全部归因于修复。',
    '', '## 源码与复核','',
    f'父版本为 auto-v3-swap `{BASE}`。`source_inventory.json`逐文件记录来源、SHA-256和继承路径；与父版本任何已有文件字节完全一致的源码不重复新增。有差异或新增的运行文件放在 `code/`，可对应到原源码的文本差异放在 `diffs/`。冻结 E1、auto-v3注册和父版本源码均未修改。',
    '在仓库根执行 `python -B experiments/scannet_result_20261020/verify.py`，核验全部归档哈希、继承源码和指标一致性。它是只读校验，不重跑模型或改写服务器结果。',
    '`external_dependencies_inventory.json`补充核对运行来源锁引用的外部第一方依赖与main R2评分文件；已有相同字节仍只引用父版本。',
    '重做分析图表：`python experiments/scannet_result_20261020/tools/analyze_scene.py --results experiments/scannet_result_20261020/results --scene scene0011_00 --output /tmp/scannet-analysis`；第二场景使用`analyze_scene0608.py`并将scene设为`scene0608_00`。需要numpy与matplotlib。',
    '完整模型重算仍需要服务器原始图像、深度、模型及环境。快照保留原绝对路径用于溯源，切勿直接运行launcher覆盖已完成输出；重跑应使用新的CVPR目录并记录路径适配。',
    'PLY保留instance_id、state和原始surface_index；三状态文件中0为黑、1为黄、3为红。每场景六个NPZ预测分别对应三方法的native/final输出。大规模中间追踪缓存和模型权重通过来源锁定位，未重复上传。','']
(PKG/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
shutil.copy2(O/'publication_diffs.md',PKG/'IMPLEMENTATION_DIFFS.md')
allfiles={p.relative_to(PKG).as_posix():dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
          for p in PKG.rglob('*') if p.is_file() and p.name not in ['archive_manifest.json']}
oversize=[(k,v['bytes']) for k,v in allfiles.items() if v['bytes']>=100*1024**2]
assert not oversize,oversize
for rel in allfiles:
    if rel.endswith(('.py','.json','.md','.txt','.patch')):
        data=(PKG/rel).read_bytes()
        assert b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----' not in data and b'-----BEGIN '+b'RSA PRIVATE KEY-----' not in data,rel
(PKG/'archive_manifest.json').write_text(json.dumps(dict(scenes=SCENES,base_commit=BASE,files=allfiles),ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(dict(scenes=SCENES,files=len(allfiles),source_files=len(inventory),
    source_reused=sum(x['disposition']=='unchanged_inherited' for x in inventory),
    external_dependencies=len(dependencies),external_dependencies_reused=sum(x['disposition']=='unchanged_inherited' for x in dependencies),
    size_MiB=sum(x['bytes'] for x in allfiles.values())/1024**2),ensure_ascii=False))
