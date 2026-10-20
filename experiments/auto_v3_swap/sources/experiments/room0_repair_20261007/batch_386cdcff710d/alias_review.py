"""Freeze a visual and geometric identity decision; never consult GT."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/data/chenkejun/CVPR/revisable_instance_map/room0_repair_20261007/batch_386cdcff710d')
LEGACY=Path('/home/chenkejun/CVPR/experiments/pilot10_repair_20261007')
sys.path.insert(0,str(LEGACY))
import run_repairs as r

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert json.loads((ROOT/'tracking_queue_complete.json').read_text())['status']=='PASS'
    diagnostics=json.loads((ROOT/'cross_seed_diagnostics.json').read_text())
    assert diagnostics['status']=='PASS' and diagnostics['GT_used'] is False
    pair=next(row for row in diagnostics['pairs'] if row['track_a']==8 and row['track_b']==11)
    a=pair['a_in_b_seed'];b=pair['b_in_a_seed']
    assert a['visible_samples']>=1000 and b['visible_samples']>=1000
    assert a['coverage']>.99 and b['coverage']>.9
    assert a['tracking_on_other_seed_frame_IoU']>.97
    assert b['tracking_on_other_seed_frame_IoU']==0
    manifest=json.loads((ROOT/'seed_manifest.json').read_text())
    cases={o['track_id']:row for row in manifest['seeds'] for o in row.get('objects',[])}
    source=r.ReplicaFrameSource(r.INPUT/'room0/configs/raw.json')
    panels=[];track_counts={}
    for oid,at_other in [(8,11),(11,8)]:
        uid=cases[oid]['case_uid'];fid=cases[at_other]['source_choice']['frame']
        path=ROOT/'cases'/uid/'tracking'/('f%06d.png'%fid)
        mask=np.array(Image.open(path))==oid;rgb=source.load_frame(fid).rgb.copy()
        rgb[mask]=(rgb[mask].astype(float)*.35+np.array([45,230,140])*.65).astype(np.uint8)
        image=Image.fromarray(rgb).resize((900,510))
        ImageDraw.Draw(image).text((12,12),f'track {oid} at f{fid}: {int(mask.sum())} pixels',fill='red',stroke_width=1,stroke_fill='white')
        panels.append(image);track_counts[str(oid)]={'other_seed_frame':fid,'foreground_pixels':int(mask.sum()),'sha256':sha(path)}
    combined=Image.new('RGB',(900,1020))
    for i,image in enumerate(panels):combined.paste(image,(0,i*510))
    combined.save(ROOT/'alias_tracking_contact.jpg',quality=93)
    review={'status':'FROZEN','GT_used':False,'aliases':{'11':8},
            'annotation_sha256':sha(ROOT/'inputs/human_selection.json'),
            'cross_seed_diagnostics_sha256':sha(ROOT/'cross_seed_diagnostics.json'),
            'review_code_sha256':sha(__file__),'tracking_contact_sha256':sha(ROOT/'alias_tracking_contact.jpg'),
            'decisions':[{'member_track_ids':[8,11],'canonical_track_id':8,
                'sources':['C0005 mask14 at f580','C0010 mask9 at f1660'],
                'physical_identity':'The same lower side table next to the white sofa; upper table track9 remains distinct',
                'visual_review':'Human-selected RGB seed contacts were inspected before any GT read; lower top plate and visible legs belong to the same lower table',
                'evidence':pair,'tracking_pixel_counts_at_other_seed':track_counts,
                'propagation_failure_acknowledged':'Track8 has zero overlap with track11 seed at f1660; track11 at f580 overlaps track8 seed with IoU above .97',
                'action':'Alias only the persistent identity. Preserve both original masks and all raw tracking. Apply unchanged per-object reliability to each frame; never fill missing masks from GT.'}],
            'other_tracks_remain_distinct':sorted(set(cases)-{8,11}),
            'no_GT_regrouping':True,'all_raw_tracks_and_numeric_thresholds_unchanged':True}
    path=ROOT/'alias_review.json'
    if path.exists():assert json.loads(path.read_text())==review
    else:path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'FROZEN','aliases':review['aliases'],'counts':track_counts,'unique_objects':29}))

if __name__=='__main__':main()
