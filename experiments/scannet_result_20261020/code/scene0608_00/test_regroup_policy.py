import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent/'grouping_r2'))
from grouping_rules import group_seeds
from regroup_policy import resolve


def alias(model, semantic=False):
    return dict(model=model, semantic_hit=semantic, touches_crop=False, crop_key='fixture',
                label=model, candidate_id=str(model))


def seed(i, count, aliases):
    return dict(id=f'M{i}', digest=str(i), frame=i, area=count, support_count=count,
                aliases=aliases, representative=aliases[0], packed=b'unchanged')


class PolicyTest(unittest.TestCase):
    def test_nonsemantic_unchanged(self):
        seeds=[seed(0,10,[alias(0)]),seed(1,8,[alias(2)])]
        inter=np.array([[10,5],[5,8]])
        result=resolve(seeds,inter,group_seeds,'fixture')
        self.assertEqual(result[3], [0,1])
        self.assertEqual(result[4:], ([],[],[]))
        np.testing.assert_array_equal(result[1],inter)

    def test_keep_sam_alias_drop_only_forbidden(self):
        seeds=[seed(0,10,[alias(2,True)]),seed(1,8,[alias(0),alias(2)]),
               seed(2,7,[alias(0)])]
        before=copy.deepcopy(seeds)
        inter=np.array([[10,6,5],[6,8,4],[5,4,7]])
        active,_,_,kept,records,dropped,_=resolve(seeds,inter,group_seeds,'fixture')
        self.assertEqual(kept,[0,1])
        self.assertEqual([s['id'] for s in dropped],['M2'])
        self.assertEqual(active[1]['aliases'],[alias(2)])
        self.assertEqual([e['entire_mask_excluded'] for e in records],[False,True])
        self.assertEqual(seeds,before)
        self.assertEqual(active[1]['packed'],before[1]['packed'])

    def test_cascade_uses_new_fixed_primary(self):
        # A owns B and S; removing CF A exposes a new S-C match.
        seeds=[seed(0,12,[alias(0)]),seed(1,10,[alias(2,True)]),
               seed(2,8,[alias(0)])]
        inter=np.array([[12,6,3],[6,10,5],[3,5,8]])
        active,new,_,kept,records,dropped,reports=resolve(seeds,inter,group_seeds,'fixture')
        self.assertEqual(kept,[1])
        self.assertEqual([r['pass_number'] for r in reports],[1,2])
        self.assertEqual([r['id'] for r in records],['M0','M2'])
        np.testing.assert_array_equal(new,[[10]])


if __name__=='__main__':
    unittest.main()
