"""Read-only verification and eight-scene selection replay for auto-v3-swap."""
from pathlib import Path
import argparse, hashlib, json, runpy, ast
ROOT=Path(__file__).resolve().parent

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise RuntimeError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-sources',action='store_true',help='Also verify deployed code, all 637 binary seed masks and external runtime locks')
    args=parser.parse_args()
    lock=read(ROOT/'source_lock.json'); config=read(ROOT/'config.json')
    for name,row in lock.items():
        p=ROOT/name;require(sha(p)==row['sha256'],'snapshot changed: '+name)
        if args.server_sources: require(sha(Path(row['source']))==row['sha256'],'server source changed: '+row['source'])
        if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8-sig'),filename=name)
    run=runpy.run_path(str(ROOT/'sources/experiments/replica8_swap34_20261020/rules/filter_rules.py'))['run_filters']
    exempt=runpy.run_path(str(ROOT/'sources/experiments/replica8_swap34_20261020/rules/semantic_exemption.py'))['apply_semantic_exemption']
    scenes=[]; total=0
    for scene,count in config['counts'].items():
        f=ROOT/'fixtures'/scene
        old=read(f/'old_selection.json'); actual=read(f/'selection.json')
        new=exempt(run(read(f/'measurements.json')),read(f/'semantic.json'))
        manual=['G017','G025'] if scene=='room2' else []
        require(actual['manual_restored_group_ids']==manual,'manual policy drift: '+scene)
        expected=sorted(set(new['retained_group_ids'])|set(manual))
        require(expected==actual['retained_group_ids'] and len(expected)==count,'retained drift: '+scene)
        require(new['stages'][:4]==actual['stages'][:4],'stage 1-4 drift: '+scene)
        require(old['stages'][:2]==actual['stages'][:2],'stage 1-2 drift: '+scene)
        require(new['semantic_pr_exempt_group_ids']==actual['semantic_pr_exempt_group_ids'],'semantic exemption drift: '+scene)
        manifest=read(f/'v1_manifest.json'); selected=read(f/'seed_manifest.json')
        original={g['id']:(i,g) for i,g in enumerate(manifest['groups'],1)}
        require(sorted(g['id'] for g in selected['groups'])==expected,'selected manifest drift: '+scene)
        for g in selected['groups']:
            i,source=original[g['id']]
            require(g['original_track_id']==i and g['primary']==source['primary'],'mask/track identity drift: '+scene)
        if args.server_sources:
            deployed=read(ROOT/'sources/experiments/replica8_swap34_20261020/config.json')[scene]
            for g in manifest['groups']:
                p=g['primary'];require(sha(Path(deployed['v1'])/p['mask'])==p['mask_sha256'],'mask drift: '+scene+'/'+g['id'])
        if scene=='room0':
            p=original['G030'][1]['primary']
            require(p['id']=='M0466' and p['mask_sha256']==config['room0_source_override']['mask_sha256'],'room0 source override drift')
        total+=len(manifest['groups']);scenes.append({'scene':scene,'v1':len(manifest['groups']),'retained':len(expected)})
    require(total==637 and sum(s['retained'] for s in scenes)==196,'total drift')
    require(config['evaluation']['frozen'] is False,'evaluation must remain development R2')
    require(sha(ROOT.parents[1]/config['evaluation']['config'])==config['evaluation']['config_sha256'],'evaluation config drift')
    if args.server_sources:
        for path,row in read(ROOT/'runtime_lock.json')['files'].items(): require(sha(Path(path))==row['sha256'],'runtime file drift: '+path)
    print(json.dumps({'status':'PASS','version':'auto-v3-swap','locked_files':len(lock),'binary_masks_verified':args.server_sources,'scenes':scenes,'v1':total,'retained':196},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
