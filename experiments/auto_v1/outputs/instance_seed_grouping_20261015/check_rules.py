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
