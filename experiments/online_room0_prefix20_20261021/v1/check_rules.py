from grouping_rules import same_instance, group_seeds

assert not same_instance(2, 5, 5), 'IoU exactly 0.25 must fail'
assert same_instance(3, 5, 5)
assert not same_instance(0, 0, 0)
assert not same_instance(0, 0, 10)
assert same_instance(1, 1, 1), 'No minimum support gate'
chain = {frozenset(('A', 'B')), frozenset(('B', 'C'))}
assert group_seeds('ABC', lambda a, b: frozenset((a, b)) in chain) == [['A', 'B'], ['C']]
star = {frozenset(('A', 'B')), frozenset(('A', 'C'))}
assert group_seeds('ABC', lambda a, b: frozenset((a, b)) in star) == [['A', 'B', 'C']]
assert group_seeds('CBA', lambda a, b: True) == [['C', 'B', 'A']]
print('PASS: strict threshold, zero support, fixed primary, no transitive expansion, no all-pairs gate')

from source_policy import eligible_aliases, assert_sam_only_groups
blocked = {'same_mask_in_another_roi': [{'ROI':'previous_roi'}]}
cf = dict(model=0, candidate_id='same_mask_in_another_roi', semantic_hit=False)
sam = dict(model=2, candidate_id='sam_window', semantic_hit=True)
generic_cf = dict(model=1, candidate_id='ordinary_chair', semantic_hit=False)
assert eligible_aliases([cf], blocked) == [], 'An alternate ROI must not revive a replaced CropFormer mask'
assert eligible_aliases([cf, sam], blocked) == [sam], 'Keep SAM when a binary mask has multiple source aliases'
assert eligible_aliases([generic_cf], blocked) == [generic_cf], 'Other objects can still use CropFormer'
assert_sam_only_groups([[0], [1]], [{'aliases':[sam]}, {'aliases':[generic_cf]}])
try:
    assert_sam_only_groups([[0,1]], [{'aliases':[sam]}, {'aliases':[cf]}])
except AssertionError:
    pass
else:
    raise AssertionError('A semantic group with CropFormer must never be exported')
print('PASS: global SAM-only source precedence and all-member export guard')

from crop_filter import artificial_crop_edges, has_crop_contact
size = (1200,680)
assert artificial_crop_edges(2,[100,20,600,500],[100,60,400,490],size)==['left']
assert artificial_crop_edges(1,[100,20,600,500],[105,60,600,500],size)==['right','bottom']
assert artificial_crop_edges(2,[0,0,1200,680],[0,0,1200,680],size)==[], 'Natural frame border is allowed'
assert artificial_crop_edges(2,[100,0,1200,680],[105,0,1200,680],size)==[], 'Only natural edges touched'
assert artificial_crop_edges(0,[0,0,1200,680],[0,0,1200,680],size)==[]
assert artificial_crop_edges(1,[100,20,600,500],[101,21,599,499],size)==[], 'One-pixel gap is not contact'
assert has_crop_contact([{'touches_crop':False},{'touches_crop':True}])
assert not has_crop_contact([{'touches_crop':False}])
print('PASS: artificial crop boundary filtering, natural-frame preservation, exact-contact boundary')
