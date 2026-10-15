"""Archive the available auto-v1 experiment snapshot; never include credentials."""
from pathlib import Path, PurePosixPath
import ast, hashlib, json, re, shutil, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor

W = Path(__file__).resolve().parents[1]
T = W/'worktrees/auto-v1/experiments/auto_v1'
STAGES = ['e1_roi_top10_20261014', 'e1_roi_top20_20261015',
    'overmerge_id_ratio_20261014', 'overmerge_id_ratio_top20_20261015',
    'vlm_threeway_20261014', 'roi_intersection_20261015',
    'mask_version_selection_20261015', 'mask_version_selection_20261015_r2',
    'semantic_sam_route_20261015', 'instance_seed_grouping_20261015',
    'instance_seed_grouping_20261015_r2', 'instance_seed_grouping_20261015_r3',
    'instance_seed_grouping_20261015_r4', 'e1_auto90_repair_20261015',
    'e1_auto90_masklocal_20261015']
PAGES = W/'results/固定案例_三模型对比_20261006'
LARGE = {
    'e1_roi_top10_20261014/room2_CUT_top10_candidates.zip': 'room2_E1_prerepair_CUT_top10_20261014',
    'e1_roi_top20_20261015/room2_CUT_top20_candidates.zip': 'room2_E1_prerepair_CUT_top20_20261015',
    'e1_auto90_repair_20261015/review.zip': 'room2_E1_auto90_repair_20261015',
}
secret = re.search(r'密码([^。\s]+)', (W/'AGENTS.md').read_text(encoding='utf-8')).group(1).encode()
patterns = [re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'), re.compile(rb'github_pat_[A-Za-z0-9_]{35,}')]
records, omissions, archives, failures = [], [], [], []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(data, name):
    if secret in data or any(p.search(data) for p in patterns):
        failures.append({'path': name, 'reason': 'credential material detected; content withheld'})
        return False
    return True


def put(data, relative, source):
    if not safe(data, relative):
        return
    if len(data) >= 100*1024*1024:
        raise ValueError('Git blob exceeds limit: '+relative)
    dest = T/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size != len(data) or dest.read_bytes() != data:
        dest.write_bytes(data)
    records.append({'path': relative, 'source': source, 'bytes': len(data), 'sha256': sha(data)})


def copy(p, rel):
    data = p.read_bytes()
    if p.suffix == '.zip':
        with zipfile.ZipFile(p) as z:
            for n in z.namelist():
                if not n.endswith('/'):
                    safe(z.read(n), rel+'!'+n)
    put(data, rel, str(p.relative_to(W)).replace('\\', '/'))


def copy_many(paths):
    # Pre-create parents serially: Windows directory creation can otherwise
    # transiently report AccessDenied when multiple threads create one folder.
    for parent in sorted({(T/p.relative_to(W)).parent for p in paths}, key=lambda p: len(p.parts)):
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        yield from pool.map(lambda p: copy(p, p.relative_to(W).as_posix()), paths)


assert subprocess.check_output(['git','-C',str(T.parents[1]),'branch','--show-current'],text=True).strip() == 'auto-v1'
T.mkdir(parents=True, exist_ok=True)
for stage in STAGES:
    root = W/'outputs'/stage
    assert root.exists(), stage
    paths = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        relative = p.relative_to(W).as_posix()
        if '__pycache__' in p.parts or p.suffix in {'.pyc','.pyo'}:
            omissions.append({'path': relative, 'reason': 'rebuildable Python cache'})
            continue
        if p.relative_to(W/'outputs').as_posix() in LARGE:
            continue
        paths.append(p)
    for _ in copy_many(paths): pass
    print('STAGE',stage,len(paths),flush=True)

for directory in sorted(PAGES.iterdir()):
    if not directory.is_dir() or not any(d in directory.name for d in ['20261014','20261015']):
        continue
    count = 0
    paths = sorted(p for p in directory.rglob('*') if p.is_file())
    for _ in copy_many(paths):
        count += 1
        if count % 5000 == 0:
            print('PAGE_PROGRESS', directory.name, count, len(paths), flush=True)
    print('PAGE',directory.name,count,flush=True)

# Large transport ZIPs are represented losslessly by their members. Reuse the
# delivered page bytes when identical; retain overwritten historical members.
for archive, page in LARGE.items():
    p = W/'outputs'/archive
    members = []
    with zipfile.ZipFile(p) as z:
        infos = [info for info in z.infolist() if not info.is_dir()]
        for info in infos:
            n=PurePosixPath(info.filename)
            assert not n.is_absolute() and '..' not in n.parts
            (T/'archive_members'/archive.split('/')[0]/Path(*n.parts)).parent.mkdir(parents=True, exist_ok=True)
        def preserve_member(info):
            n = PurePosixPath(info.filename)
            assert not n.is_absolute() and '..' not in n.parts
            data = z.read(info.filename)
            rel = (PAGES/page/Path(*n.parts)).relative_to(W).as_posix()
            dest = T/rel
            if not dest.is_file() or sha(dest.read_bytes()) != sha(data):
                rel = 'archive_members/'+archive.split('/')[0]+'/'+n.as_posix()
                put(data, rel, 'outputs/'+archive+'!'+n.as_posix())
            return {'member':n.as_posix(),'stored_as':rel,'bytes':len(data),'sha256':sha(data)}
        with ThreadPoolExecutor(max_workers=16) as pool:
            members=list(pool.map(preserve_member,infos))
    archives.append({'source':'outputs/'+archive,'bytes':p.stat().st_size,
        'original_zip_sha256':sha(p.read_bytes()),'members':members,
        'storage':'Every original member preserved; original ZIP transport container omitted'})
    print('ARCHIVE_MEMBERS_VERIFIED',archive,len(members),flush=True)

copy(W/'outputs/replica_gt_scope_20261008/ssh_task.py', 'outputs/replica_gt_scope_20261008/ssh_task.py')
copy(Path(__file__), 'package_auto_v1.py')
copy(W/'outputs/verify_auto_v1.py', 'verify_auto_v1.py')
records.sort(key=lambda r: r['path'])
receipt = {'status':'BLOCKED' if failures else 'PASS',
    'branch':'auto-v1','base_main':subprocess.check_output(['git','-C',str(T.parents[1]),'rev-parse','HEAD'],text=True).strip(),
    'scope':'Available local auto pipeline implementations, all delivered results/pages, full final instance PLY, and unfinished mask-local draft',
    'server_only_artifacts':'Not downloaded in this publication: full 2000-frame native tracking volumes, vote ledgers, and native TSDF NPZ files remain at their recorded server paths/hashes. Server SSH unavailable.',
    'files':records,'omissions':omissions,'archive_member_storage':archives,
    'credential_scan_failures':failures,'file_count':len(records),'bytes':sum(r['bytes'] for r in records)}
(T/'publication_manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'files':len(records),'bytes':receipt['bytes'],'credential_scan_failures':failures},ensure_ascii=False),flush=True)
assert not failures, 'Do not publish until credential scan passes'
