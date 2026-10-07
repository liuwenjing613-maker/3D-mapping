"""GT-free reciprocal seed visibility and cross-source tracking correspondence."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY));sys.path.insert(0,str(LEGACY/'strict_local_repair_20261007'));sys.path.insert(0,str(LEGACY/'objectwise_repair_20261007'))
import run_repairs as r
import run_case as c
import fixed_surface_repair as f
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room2_repair_20261007/batch_8e890bff216d')

def main():
    scene=r.Scene('room2'); policy=json.loads((LEGACY/'objectwise_repair_20261007/policy.json').read_text())
    cases=json.loads((ROOT/'seed_diagnostics.json').read_text())['cases']
    records={};frames={};seeds={}
    for case in cases:
        uid=case['case_uid']; fid=case['source_choice']['frame']; frames[uid]=scene.source.load_frame(fid)
        seeds[uid]=np.array(Image.open(ROOT/'cases'/uid/'prepared_seed.png'))
        z=f.load_arrays(ROOT/'cases'/uid/'seed_projected_support.npz')
        for obj in case['objects']:
            records[obj['track_id']]={**obj,'case_uid':uid,'frame':fid,'points':np.unique(z['surface_point_index'][z['track_id']==obj['track_id']])}
    rows=[]
    for a,ra in records.items():
        for b,rb in records.items():
            if a>=b or ra['case_uid']==rb['case_uid']: continue
            row={'track_a':a,'track_b':b,'same_old_family':ra['old_family_id']==rb['old_family_id']}
            for source,target,key in [(ra,rb,'a_in_b_seed'),(rb,ra,'b_in_a_seed')]:
                u,v=c.visible_seed(scene,frames[target['case_uid']],source['points'],policy)
                row[key]={'visible_samples':len(u),'coverage':float(np.mean(seeds[target['case_uid']][v,u]==target['track_id'])) if len(u) else None}
                othertrack=ROOT/'cases'/target['case_uid']/'tracking'/('f%06d.png'%source['frame'])
                if othertrack.exists():
                    candidate=np.array(Image.open(othertrack))==target['track_id']; own=seeds[source['case_uid']]==source['track_id']
                    row[key]['tracking_on_other_seed_frame_IoU']=float(np.sum(candidate&own)/max(1,np.sum(candidate|own)))
                    row[key]['tracking_on_other_seed_frame_seed_coverage']=float(np.sum(candidate&own)/max(1,np.sum(own)))
            if row['same_old_family'] or any((row[k]['coverage'] or 0)>.1 for k in ['a_in_b_seed','b_in_a_seed']): rows.append(row)
    f.atomic_json(ROOT/'cross_seed_diagnostics.json',{'status':'PASS','GT_used':False,'pairs':rows})
    print(json.dumps(rows,ensure_ascii=False))

if __name__=='__main__': main()
