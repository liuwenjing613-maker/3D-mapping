import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from instance_repair.seed_boundary import boundary_evidence, screen, validate_selected, sha

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT/'experiments/auto_v3_swap/sources/experiments/auto_v2_room0_20261019/v1'


def alias(model=0, box=None):
    return dict(model=model, source_box=box or [0,0,1200,680], semantic_hit=False)


class BoundaryTests(unittest.TestCase):
    def test_each_frame_edge_and_corner(self):
        for y,x in [(0,600),(679,600),(340,0),(340,1199),(679,0)]:
            a=np.zeros((680,1200),bool);a[y,x]=True
            self.assertTrue(boundary_evidence(a,[alias()])['touches_any_edge'])

    def test_one_pixel_inset_is_allowed_without_eroding(self):
        a=np.zeros((680,1200),bool);a[1:-1,1:-1]=True;b=a.copy()
        self.assertFalse(boundary_evidence(a,[alias()])['touches_any_edge'])
        np.testing.assert_array_equal(a,b)

    def test_every_artificial_edge_half_open_coordinates(self):
        for y,x in [(100,120),(199,120),(150,100),(150,199)]:
            a=np.zeros((680,1200),bool);a[y,x]=True
            d=boundary_evidence(a,[alias(1,[100,100,200,200])])
            self.assertFalse(d['touches_full_frame']);self.assertTrue(d['crop_contacts'])

    def test_clean_alias_cannot_hide_other_crop_contact(self):
        a=np.zeros((680,1200),bool);a[150,100]=True
        self.assertTrue(boundary_evidence(a,[alias(),alias(2,[100,100,200,200])])['touches_any_edge'])

    def test_original_frame_boundary_also_rejected_for_crop_source(self):
        a=np.zeros((680,1200),bool);a[660:680,0:11]=True
        d=boundary_evidence(a,[alias(1,[0,352,1200,680])])
        self.assertTrue(d['touches_full_frame']);self.assertFalse(d['crop_contacts'])

    def test_bad_provenance_and_labels_fail_closed(self):
        a=np.zeros((680,1200),np.uint16);a[300,300]=1
        for aliases in [[],[{}],[alias(3)],[alias(0,[1,1,1200,680])],[alias(1,[-1,0,1200,680])]]:
            with self.assertRaises(ValueError):boundary_evidence(a,aliases)
        a[300,300]=90
        with self.assertRaises(ValueError):boundary_evidence(a,[alias()])

    def test_regroup_before_primary_selection_and_guard_auxiliary(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'masks').mkdir();rows=[]
            arrays=[np.array(x,dtype=np.int32) for x in [[0,1,2,3,4],[1,2,3,4],[3,4]]]
            for i,ix in enumerate(arrays):
                a=np.zeros((680,1200),bool);a[100:120,0:20] = i==0
                if i:a[100:120,100+i*30:120+i*30]=True
                path=p/'masks'/f'{i}.png';Image.fromarray(a.astype(np.uint8)*255).save(path)
                rows.append(dict(index=i,id=f'M{i}',frame=i*5,mask=f'masks/{i}.png',mask_sha256=sha(path),
                    area=int(a.sum()),digest=hashlib.sha256((i*5).to_bytes(4,'little')+np.packbits(a.ravel()).tobytes()).hexdigest(),
                    aliases=[alias()],representative=alias(),support_count=len(ix),support_sha256=hashlib.sha256(ix.tobytes()).hexdigest()))
            (p/'grouping.json').write_text(json.dumps(dict(seeds=rows)))
            np.savez(p/'supports.npz',indices=np.concatenate(arrays),offsets=np.r_[0,np.cumsum([len(x) for x in arrays])])
            inter=np.array([[len(np.intersect1d(a,b)) for b in arrays] for a in arrays])
            np.savez(p/'pairs.npz',intersection=inter)
            before={str(x):sha(x) for x in p.rglob('*') if x.is_file()}
            stats=screen(p/'grouping.json',p/'supports.npz',p/'pairs.npz',RULES,p/'out')
            self.assertEqual(stats['total_edge_removed'],1);self.assertEqual(stats['groups'],1)
            m=json.loads((p/'out/seed_manifest.json').read_text())
            self.assertEqual(m['groups'][0]['primary']['id'],'M1');self.assertEqual(m['groups'][0]['auxiliary_ids'],['M2'])
            with self.assertRaises(ValueError):validate_selected(p/'out/seed_manifest.json',p/'out/grouping.json',p)
            m['status']='complete';(p/'selected.json').write_text(json.dumps(m))
            self.assertEqual(validate_selected(p/'selected.json',p/'out/grouping.json',p)['checked_masks'],2)
            m['groups'][0]['auxiliary_ids'].append('M0');(p/'selected.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):validate_selected(p/'selected.json',p/'out/grouping.json',p)
            self.assertEqual(before,{path:sha(path) for path in before})
            with self.assertRaises(ValueError):screen(p/'grouping.json',p/'supports.npz',p/'pairs.npz',RULES,p/'out')

    def test_all_candidates_excluded_is_valid_empty_output(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=np.zeros((680,1200),bool);a[100:120,:20]=True
            Image.fromarray(a.astype(np.uint8)*255).save(p/'mask.png')
            ix=np.array([1,2,3],np.int32)
            row=dict(index=0,id='M0',frame=0,mask='mask.png',mask_sha256=sha(p/'mask.png'),
                area=400,digest=hashlib.sha256(bytes(4)+np.packbits(a.ravel()).tobytes()).hexdigest(),
                aliases=[alias()],support_count=3,support_sha256=hashlib.sha256(ix.tobytes()).hexdigest())
            (p/'grouping.json').write_text(json.dumps(dict(seeds=[row])))
            np.savez(p/'supports.npz',indices=ix,offsets=np.array([0,3]))
            np.savez(p/'pairs.npz',intersection=np.array([[3]]))
            stats=screen(p/'grouping.json',p/'supports.npz',p/'pairs.npz',RULES,p/'out')
            self.assertEqual(stats['total_edge_removed'],1);self.assertEqual(stats['groups'],0)
            with np.load(p/'out/pair_metrics.npz') as z:self.assertEqual(z['intersection'].shape,(0,0))
            with np.load(p/'out/surface_supports.npz') as z:
                self.assertEqual(len(z['indices']),0);np.testing.assert_array_equal(z['offsets'],[0])
            self.assertEqual(json.loads((p/'out/seed_manifest.json').read_text())['groups'],[])


if __name__=='__main__':unittest.main()
