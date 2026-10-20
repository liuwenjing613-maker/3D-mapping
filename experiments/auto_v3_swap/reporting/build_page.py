from pathlib import Path
import csv,hashlib,json,os,shutil
W=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
SITE=W/'results/固定案例_三模型对比_20261006/replica8_auto_E1_20261019'
OUT=SITE/'swap34';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=read(O/'delivery/comparison.json')
def add_file(scene,src,info,variant):
    target=OUT/'ply'/scene['scene']/src.name;target.parent.mkdir(parents=True,exist_ok=True)
    assert sha(src)==info['sha256']
    if not target.exists():
        try:os.link(src,target)
        except OSError:shutil.copyfile(src,target)
    assert sha(target)==info['sha256']
    scene['ply'].append(dict(variant=variant,kind=info.get('kind') or ('instances' if '_instances' in src.name else 'three_states_black_yellow_red'),
        file=src.name,url=target.relative_to(SITE).as_posix(),sha256=info['sha256'],
        vertices=info.get('vertices',info.get('points'))))
for scene in data['scenes']:
    name=scene['scene'];old=read(SITE/'filtering'/name/'data.json')
    scene['old']=old;scene['ply']=[]
    assert len(scene['selection']['records'])==len(old['rows'])
    for row in old['rows']:
        for img in row['images'].values():assert (SITE/img).is_file(),img
    if name!='room1':
        root=W/'outputs/replica7_ply_20261020'/name;m=read(root/'manifest.json')
        for filename,info in m['files'].items():
            variant='baseline' if info['variant']=='no_repair' else 'old_partial' if 'partial' in info['variant'] else 'old_E1'
            add_file(scene,root/filename,info,variant)
            if scene['action']=='reuse_identical_selection' and variant!='baseline':
                assert scene['source_locks'][variant+'_final']['surface_sha256']==scene['source_locks'][variant.replace('old','new')+'_final']['surface_sha256']
                add_file(scene,root/filename,info,variant.replace('old','new'))
    else:
        for folder in ['room1_E1_ply_20261020','room1_partial_E1_ply_20261020']:
            root=W/'outputs'/folder/'ply';m=read(root/'manifest.json')
            for filename,info in m['files'].items():
                variant='baseline' if 'no_repair' in filename else 'old_partial' if 'partial' in filename else 'old_E1'
                add_file(scene,root/filename,info,variant)
        root=W/'outputs/room1_swap34_20261020/delivery/ply';m=read(root/'manifest.json')
        for info in m['exports']:add_file(scene,root/info['file'],info,'new_partial' if 'partial' in info['variant'] else 'new_E1')
    root=O/'delivery'/name/'ply'
    if (root/'manifest.json').exists():
        for info in read(root/'manifest.json')['exports']:
            variant='new_partial' if 'partial' in info['variant'] else 'new_E1'
            assert info['source_sha256']==scene['source_locks'][variant+'_final']['surface_sha256']
            add_file(scene,root/info['file'],info,variant)
    for link in scene['ply']:
        if link['kind'].startswith('three_states'):
            row=next(x for x in scene['rows'] if x['variant']==link['variant'] and x['stage']=='final')
            assert link['vertices']==row['gray_total'],(name,link)
    reports=OUT/'reports'/name;reports.mkdir(parents=True,exist_ok=True)
    for key in ['selection','order','association','source_locks','tracking_difference_audit']:
        (reports/(key+'.json')).write_text(json.dumps(scene[key],ensure_ascii=False,indent=2),encoding='utf-8')
assert sum(len(s['old']['rows']) for s in data['scenes'])==637
assert sum(len(s['selection']['retained_group_ids']) for s in data['scenes'])==196
(OUT/'data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    columns=['scene','variant','stage','F1','PQ','AP','AP50','mCov','TP','FP','FN','gray_total','selected']
    writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
    for s in data['scenes']:
        for row in s['rows']:writer.writerow(dict(row,scene=s['scene']))
for name in ['style.css','app.js']:shutil.copyfile(O/'web'/name,OUT/name)
shutil.copyfile(O/'web/index.html',SITE/'replica8_swap34.html')
p=SITE/'index.html';t=p.read_text(encoding='utf-8')
banner='<section class="panel" id="swap34Link"><h2>八场景新顺序实验</h2><p>包含筛选提前到覆盖充分判定之前，201 → 196 张；查看原版 E1、partial、无修复的统一 R2 对比及全部筛选过程。</p><a href="replica8_swap34.html">打开八场景新顺序结果与 PLY →</a></section>'
if 'id="swap34Link"' not in t:p.write_text(t.replace('<main>','<main>'+banner),encoding='utf-8')
print('PAGE_BUILT',data['status'],637,196,flush=True)
