"""Publish only the completed room0 experiment to the authorized SAM-track branch."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time,os,signal
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d')
CODE=Path('/home/chenkejun/CVPR/experiments/room0_repair_20261007/batch_386cdcff710d')
WORK=Path('/home/chenkejun/CVPR/worktrees/sam-track')
PRIMARY=Path('/home/chenkejun/CVPR/3D-mapping')
EXPECTED='a36dc6f3fa33c16a065b5e4e67659b57358667b7'
PREFIX='experiments/sam_track/room0_batch_repair_20261007'
DOC='experiments/sam_track/docs/room0_batch_changes_20261007.md'

def git(repo,*args):
    for attempt in range(3 if args[0]=='ls-remote' else 1):
        process=subprocess.Popen(['git','-C',str(repo),'-c','http.version=HTTP/1.1',
                '-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=20',*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:stdout,stderr=process.communicate(timeout=35 if args[0]=='ls-remote' else 60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);stdout,stderr=process.communicate();stderr+='\nGit command reached the bounded timeout.'
        if process.returncode==0:return stdout.rstrip('\n')
        if args[0]=='ls-remote' and attempt<2:time.sleep(2*(attempt+1))
    raise RuntimeError(stderr[-1000:])

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def state(repo):
    paths=set()
    for args in [('diff','--name-only','-z'),('diff','--cached','--name-only','-z'),('ls-files','--others','--exclude-standard','-z')]:
        paths.update(p for p in git(repo,*args).split('\0') if p)
    return {'head':git(repo,'rev-parse','HEAD'),'status':git(repo,'status','--porcelain=v1'),
            'local_files':{p:sha(repo/p) if (repo/p).is_file() else 'missing' for p in paths}}

def main():
    assert not (ROOT/'branch_publication.json').exists()
    final=json.loads((ROOT/'final_validation.json').read_text());assert final['status']=='PASS'
    ev=json.loads((ROOT/'evaluation_summary.json').read_text());assert ev['status']=='PASS'
    assert json.loads((ROOT/'review_bundle.json').read_text())['PLY_roundtrip_verified']
    assert git(WORK,'rev-parse','HEAD')==EXPECTED
    assert git(WORK,'rev-parse','--abbrev-ref','HEAD')=='SAM-track'
    assert git(WORK,'ls-remote','origin','refs/heads/SAM-track').split()[0]==EXPECTED
    assert git(WORK,'status','--porcelain=v1')==''
    primary=state(PRIMARY)
    folder=WORK/PREFIX;assert not folder.exists();folder.mkdir(parents=True)
    names=['prepare_local.py','prepare_server.py','cross_seed_diagnostics.py','alias_review.py',
           'joint_batch_ops.py','test_joint_batch_ops.py','tracking_queue.py','run_batch.py','pipeline.py',
           'evaluate_batch.py','export_review.py','export_review_v2.py','export_recovery.py',
           'build_review.py','room0_viewer.js','CHANGES_CN.md','README_CN.md']
    names+=sorted(p.name for p in CODE.glob('room0-*.json'))
    assert len([name for name in names if name.startswith('room0-') and name.endswith('.json')])==12
    source_hashes={}
    for name in names:
        shutil.copyfile(CODE/name,folder/name);source_hashes[name]=sha(CODE/name)
        assert source_hashes[name]==sha(folder/name)
    shutil.copyfile(ROOT/'inputs/human_selection.json',folder/'human_selection.json')
    source_hashes['human_selection.json']=sha(ROOT/'inputs/human_selection.json')
    shutil.copyfile(ROOT/'alias_review.json',folder/'alias_review.json');source_hashes['alias_review.json']=sha(ROOT/'alias_review.json')
    shutil.copyfile(CODE/'CHANGES_CN.md',WORK/DOC)
    rep=json.loads((ROOT/'validated/full_track/complete.json').read_text());m=ev['comparisons']['full_track']
    manifest={'status':'PASS','runtime_source':str(CODE),'data_root':str(ROOT),
      'annotation_sha256':source_hashes['human_selection.json'],'source_sha256':source_hashes,
      'final_validation':final,'alias_decisions':json.loads((ROOT/'alias_review.json').read_text()),
      'result_summary':{'baseline':ev['baseline_v3_metrics'],'full_track':m,
         'final_changed_points':rep['strict_commit']['final_inside_changes'],
         'gray_point_delta':rep['whole_scene_unknown_after']-rep['whole_scene_unknown_before'],
         'outside_changed_points':rep['strict_commit']['final_outside_changes']},
      'prediction_freeze_sha256':sha(ROOT/'evaluation_freeze.json'),
      'formal_benchmark_result':False,'no_weights_datasets_PLY_or_secrets_committed':True}
    (folder/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    git(WORK,'add','--',PREFIX,DOC)
    staged=git(WORK,'diff','--cached','--name-only').splitlines()
    wanted={PREFIX+'/'+name for name in source_hashes}|{PREFIX+'/source_manifest.json',DOC}
    assert set(staged)==wanted
    for path in staged:
        data=subprocess.check_output(['git','-C',str(WORK),'show',':'+path])
        assert data==(WORK/path).read_bytes(),path
        assert len(data)<8*1024*1024
    for name,digest in source_hashes.items():
        origin=ROOT/'inputs/human_selection.json' if name=='human_selection.json' else ROOT/'alias_review.json' if name=='alias_review.json' else CODE/name
        assert sha(origin)==digest==sha(folder/name)
    git(WORK,'diff','--cached','--check')
    git(WORK,'commit','-m','Add validated room0 tracking repairs from human selections')
    commit=git(WORK,'rev-parse','HEAD')
    push=subprocess.run(['git','-C',str(WORK),'-c','http.version=HTTP/1.1','push','origin','HEAD:refs/heads/SAM-track'],capture_output=True,text=True)
    print(json.dumps({'push_exit':push.returncode,'stdout':push.stdout[-900:],'stderr':push.stderr[-900:]}),flush=True)
    assert git(WORK,'ls-remote','origin','refs/heads/SAM-track').split()[0]==commit
    assert git(WORK,'status','--porcelain=v1')==''
    assert state(PRIMARY)==primary
    receipt={'status':'PASS','branch':'SAM-track','commit':commit,'remote_commit':commit,'files':staged,
       'primary_checkout_and_local_changes_preserved':True,'runtime_sources_byte_identical':True,'force_push':False}
    (ROOT/'branch_publication.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
