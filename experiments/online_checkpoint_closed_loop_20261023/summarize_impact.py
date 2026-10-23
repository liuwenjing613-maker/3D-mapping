"""Summarize measured downstream differences without changing either run."""
from pathlib import Path
import json

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')

def read(path):return json.loads(path.read_text())

def main():
    result={}
    for name in ['tracking_comparison','repair_comparison','fixed_tracking_comparison','fixed_repair_comparison']:
        path=OUT/(name+'.json')
        if not path.exists():continue
        obj=read(path);checks=obj['checks']
        result[name]=dict(status=obj['status'],comparisons=len(checks),
            required_differences=sum(not r['exact'] for r in checks if r.get('required',True)))
        if name=='tracking_comparison':
            result[name]['key_checks']=[r for r in checks if r['name'] in ['dense_SAM2_label_pixels',
                'Gate_accept_reject_decisions','Gate_accepted_mask_pixels','repair_persistent_ID_objects']]
        elif name=='repair_comparison':
            selected=[]
            for row in checks:
                n=row['name']
                if n.endswith('/instance_id') or n.endswith('/state') or n.endswith('/allowed_surface_ids') or n.startswith('commit/') or n.startswith('repair/e1/'):
                    out=dict(row)
                    if 'differences' in out:
                        diff=out.pop('differences')
                        out.update(differing_files=len(diff),differing_frames=len(set(r['frame'] for r in diff)),
                            total_different_pixels=sum(r['different_pixels'] for r in diff),detail=diff)
                    selected.append(out)
                if n=='repair/per_frame_transactions':
                    diff=row['differences']
                    selected.append(dict(name=n,exact=row['exact'],comparisons=row['comparisons'],
                        differing_fields=len(diff),differing_frames=len(set(r['frame'] for r in diff))))
            result[name]['key_checks']=selected
    for mode in ['reference','checkpoint']:
        path=OUT/mode/'repair/commit_complete.json'
        if path.exists():
            c=read(path)
            result[mode+'_commit']={k:c[k] for k in ['status','processed_frames','last_raw_frame','map_version',
                'next_instance_id','checkpoint_sha256','identity_counts_sha256','source_pixel_support_sha256','final_map_sha256']}
    audit=OUT/'final_audit.json'
    if audit.exists():result['frame201']=read(audit)
    (OUT/'impact_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
