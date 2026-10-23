"""Check committed-state inheritance, immutable inputs, and causal read audits."""
from pathlib import Path
import hashlib,json
import numpy as np

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    inheritance=[];inputs=[];access=[];new_frame_decisions=[];prior_commits=[]
    for mode in ['reference','checkpoint']:
        root=OUT/mode;commit=read(root/'repair/commit_complete.json');probe=root/'frame201/mapping'
        audit=read(probe/'frame201_state_audit.json');restore=read(probe/'restore_audit.json');done=read(probe/'mapping_complete.json')
        assert commit['status']==audit['status']==restore['status']==done['status']=='PASS'
        assert audit['checkpoint_sha256']==commit['checkpoint_sha256']==sha(commit['checkpoint'])
        assert audit['restored_map_version']==audit['map_version_used_before_association']==commit['map_version']
        assert audit['frame_id']==1000 and audit['repaired_state_used']
        assert done['frame_count']==201 and done['last_frame_id']==1000 and done['map_version']==commit['map_version']+1
        assert done['historical_committed_entries_unchanged'] and done['all_prior_layers_retained']
        assert restore['all_source_pixel_samples_exact'] and restore['all_journal_PIDs_exact']
        rows=[json.loads(s) for s in (probe/'association/associations.jsonl').read_text().splitlines()]
        current=[r for r in rows if r['frame_id']==1000]
        new_frame_decisions.append(current)
        inheritance.append(dict(mode=mode,status='PASS',processed_frame=201,raw_frame=1000,
            new_commit_checkpoint_sha256=commit['checkpoint_sha256'],map_version_before=commit['map_version'],
            map_version_after=done['map_version'],frame201_observations=len(current),restored_entries=restore['restored_entries'],
            journal_pixel_and_PID_restore_exact=True,all_committed_entries_preserved_after_frame201=True))
        freeze=read(root/'candidates/mapping_input_freeze.json')['source_hashes']
        for p,h in freeze.items():assert sha(p)==h,p
        prior=read(root/'mapping/prior_commit.json')
        assert sha(prior['prior_commit_path'])==prior['prior_commit_sha256']
        prior_commits.append((str(Path(prior['prior_commit_path']).resolve()),prior['prior_commit_sha256']))
        inputs.append(dict(mode=mode,status='PASS',mapping_files_verified=len(freeze),prior100_commit_unchanged=True))
    assert prior_commits[0]==prior_commits[1]
    a,b=[OUT/m/'frame201/mapping' for m in ['reference','checkpoint']]
    rows=[]
    for rel in ['association/associations.jsonl','association/observations.jsonl']:
        rows.append(dict(file=rel,exact=(a/rel).read_bytes()==(b/rel).read_bytes()))
    for rel in ['association/observation_support_3cm.npz','association/identity_counts.npz']:
        x,y=load(a/rel),load(b/rel)
        assert sorted(x)==sorted(y)
        for key in x:
            rows.append(dict(file=rel,key=key,exact=x[key].dtype==y[key].dtype and np.array_equal(x[key],y[key])))
    for mode in ['reference','checkpoint','fixed_reference','fixed_checkpoint']:
        roots=[OUT/mode]
        if mode in ['reference','checkpoint']:roots.append(OUT/mode/'frame201')
        for root in roots:
            count=0;violations=[]
            for path in (root/'io_audit').glob('*.jsonl'):
                with path.open() as f:
                    for line in f:
                        row=json.loads(line)
                        if row.get('probe'):continue
                        count+=1
                        if row.get('blocked'):violations.append(row)
            access.append(dict(root=str(root),recorded_nonprobe_opens=count,blocked_nonprobe_opens=len(violations),violations=violations[:10]))
    result=dict(status='PASS' if all(r['exact'] for r in rows) and not any(r['blocked_nonprobe_opens'] for r in access) else 'DIFFERENT',
        each_mode_inherits_its_own_new_commit=True,
        same_starting_frame100_commit=True,starting_frame100_commit_path=prior_commits[0][0],starting_frame100_commit_sha256=prior_commits[0][1],
        frame201_new_association_decisions_exact=new_frame_decisions[0]==new_frame_decisions[1],
        inheritance=inheritance,frame201_equivalence=rows,immutable_inputs=inputs,causal_access_audit=access)
    (OUT/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
