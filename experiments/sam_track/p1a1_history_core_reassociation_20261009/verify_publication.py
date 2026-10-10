"""Verify archived measured bytes and current R2 source without private data."""
from pathlib import Path
import hashlib,json,zipfile

HERE=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def main():
    provenance=read(HERE/'publication_provenance.json')
    for path,digest in provenance['original_artifact_sha256'].items():
        assert sha(REPO/path)==digest,path
    package=HERE/'results/two_scene_history_reassociation_audit.zip'
    assert sha(package)==provenance['source_bundle_sha256']
    with zipfile.ZipFile(package) as z:
        manifest=json.loads(z.read('bundle_manifest.json'))
        assert z.testzip() is None
        for name,row in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==row['sha256'],name
    config=read(REPO/'unified_eval/configs/current_protocol.json')
    assert config['profile_revision']==2 and config['evaluable_GT_count']==350 and config['frozen'] is False
    assert sha(REPO/'unified_eval/configs'/config['config_file'])==config['config_sha256']==provenance['config_sha256']
    for name,digest in config['evaluator_code_sha256'].items():assert sha(REPO/'unified_eval'/name)==digest,name
    done=read(HERE/'results/evaluation_r2/evaluation_complete.json')
    assert sha(HERE/'results/evaluation_r2/comparison.json')==done['comparison_sha256']
    result=read(HERE/'results/evaluation_r2/comparison.json')
    base=read(REPO/'experiments/sam_track/p1a1_strict_vote_20261009/baseline_materialization/results/summary_r2.json')
    for scene in ('room0','room2'):
        for stage in ('native','final'):
            key='strict_baseline_'+stage
            assert base['per_scene'][scene][key]==result['per_scene'][scene][key]
            assert base['states'][scene][key]==result['states'][scene][key]
    clouds=read(HERE/'exports/pointcloud_manifest.json')
    assert len(clouds['files'])==6 and all(row['server_file'] for row in clouds['files'])
    print(json.dumps({'status':'PASS','measured_files_verified':len(provenance['original_artifact_sha256']),
        'archive_members_verified':len(manifest['files'])+1,'current_R2_source_lock':True,
        'baseline_R2_scores_match':True,'six_pointclouds_referenced':True}))

if __name__=='__main__':main()
