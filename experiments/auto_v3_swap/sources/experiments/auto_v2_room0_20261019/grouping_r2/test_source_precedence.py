import unittest
import numpy as np
from source_policy import mixed_group_source_exclusions,assert_sam_only_groups

def seed(name,model,hit=False):
 return dict(id=name,support_count=100,aliases=[dict(model=model,semantic_hit=hit)])

class SourcePrecedenceTests(unittest.TestCase):
 def test_ordinary_group_unchanged(self):
  s=[seed('a',0),seed('b',2)]
  self.assertEqual(mixed_group_source_exclusions([[0,1]],s,np.array([[100,80],[80,100]]),1),{})
 def test_mixed_source_group_keeps_sam_and_records_direct_evidence(self):
  s=[seed('a',0),seed('b',2,True),seed('c',1)]
  x=mixed_group_source_exclusions([[0,1],[2]],s,np.full((3,3),80),1)
  self.assertEqual(set(x),{0});self.assertEqual(x[0]['retained_aliases'],[])
  self.assertEqual(x[0]['semantic_SAM_mask_ids'],['b'])
  self.assertEqual(x[0]['primary_edges'][0]['intersection'],80)
 def test_binary_duplicate_retains_sam_alias(self):
  s=[seed('a',0),seed('b',2,True)];s[0]['aliases'].append(dict(model=2,semantic_hit=False))
  x=mixed_group_source_exclusions([[0,1]],s,np.full((2,2),80),1)
  self.assertEqual(x[0]['retained_aliases'],[dict(model=2,semantic_hit=False)])
 def test_sam_only_guard_still_rejects_unresolved_groups(self):
  with self.assertRaises(AssertionError):assert_sam_only_groups([[0,1]],[seed('a',0),seed('b',2,True)])
if __name__=='__main__':unittest.main()
