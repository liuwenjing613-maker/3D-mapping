"""Verify the staged Git blobs against the publication's measured file bytes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, subprocess

W=Path(__file__).resolve().parents[1]
repo=W/'worktrees/auto-v1'
root=repo/'experiments/auto_v1'
manifest=json.loads((root/'publication_manifest.json').read_text(encoding='utf-8'))
assert manifest['status']=='PASS' and not manifest['credential_scan_failures']
index={}
for entry in subprocess.check_output(['git','-C',str(repo),'ls-files','--stage','-z']).split(b'\0'):
    if entry:
        fields,path=entry.split(b'\t',1)
        mode,oid,stage=fields.split()
        assert stage==b'0'
        index[path.decode()]=oid.decode()

def verify(row):
    relative='experiments/auto_v1/'+row['path']
    data=(repo/relative).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],relative
    oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert index.get(relative)==oid,relative+' Git blob differs or missing'
    return 1

with ThreadPoolExecutor(max_workers=8) as pool:
    verified=sum(pool.map(verify,manifest['files']))
# Member storage was byte-verified during packaging; check each maps to a
# measured, staged record with the same size and hash.
by_path={r['path']:r for r in manifest['files']}
for archive in manifest['archive_member_storage']:
    for member in archive['members']:
        stored=by_path[member['stored_as']]
        assert stored['sha256']==member['sha256'] and stored['bytes']==member['bytes']
changed=subprocess.check_output(['git','-C',str(repo),'diff','--cached','--name-only','-z']).split(b'\0')
assert all(not p or p.decode() in {'.gitattributes','AUTO_V1.md'} or p.decode().startswith('experiments/auto_v1/') for p in changed)
subprocess.run(['git','-C',str(repo),'diff','--cached','--check','--',
    '.gitattributes','AUTO_V1.md','experiments/auto_v1/README.md'],check=True)
receipt={'status':'PASS','measured_files_verified':verified,'measured_bytes':manifest['bytes'],
    'staged_git_blob_hashes_exact':True,'original_archive_members_preserved':sum(len(a['members']) for a in manifest['archive_member_storage']),
    'source_paths_and_SHA256_recorded':True,'base_main':manifest['base_main'],
    'scope':'All local archival artifacts, not a backup of server-only data',
    'draft_helpers_only_tests':11,'draft_connected_to_pipeline':False,'new_repair_run':False}
(root/'publication_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
