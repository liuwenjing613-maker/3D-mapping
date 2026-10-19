import unittest
import numpy as np
from rules import compare, decide


class ContainmentTests(unittest.TestCase):
    def test_strict_70_boundary(self):
        small = np.arange(10)
        at70 = np.r_[np.arange(7), np.arange(20, 33)]
        above70 = np.r_[np.arange(8), np.arange(20, 32)]
        self.assertFalse(compare('S', small, 'L', at70)['passes'])
        self.assertTrue(compare('S', small, 'L', above70)['passes'])

    def test_containment_not_iou_or_pair_visible_denominator(self):
        small, large = np.arange(100), np.arange(50, 1050)
        self.assertFalse(compare('S', small, 'L', large)['passes'])
        small = np.arange(50, 150)
        p = compare('S', small, 'L', large)
        self.assertTrue(p['passes'])
        self.assertEqual(p['containment'], 1)
        self.assertEqual(p['original_iou'], .1)
        reverse = compare('L', large, 'S', small)
        self.assertEqual((p['small'], p['large'], p['containment']),
                         (reverse['small'], reverse['large'], reverse['containment']))

    def test_direct_pair_decisions_independent_of_order(self):
        a = np.arange(100)
        b = np.r_[np.arange(71), np.arange(100, 229)]
        c = np.r_[np.arange(14), np.arange(100, 229), np.arange(300, 457)]
        ab, bc, ac = compare('A', a, 'B', b), compare('B', b, 'C', c), compare('A', a, 'C', c)
        self.assertTrue(ab['passes'] and bc['passes'])
        self.assertFalse(ac['passes'])
        self.assertEqual(decide(['A', 'B', 'C'], [ab, bc, ac]), (['A', 'B'], ['C']))
        self.assertEqual(decide(['C', 'B', 'A'], [ac, bc, ab]), (['A', 'B'], ['C']))

    def test_equal_size_or_duplicate_indices_cannot_silently_choose(self):
        with self.assertRaises(ValueError):
            compare('A', np.arange(10), 'B', np.arange(10))
        with self.assertRaises(ValueError):
            compare('A', np.array([1, 1, 2]), 'B', np.arange(10))


if __name__ == '__main__':
    unittest.main()
