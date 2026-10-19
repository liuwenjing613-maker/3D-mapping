import unittest
from rules import excluded


class StrictThresholdTests(unittest.TestCase):
    def test_p_boundary_and_otherwise_low_r(self):
        self.assertFalse(excluded(10, 100, 100))
        self.assertTrue(excluded(11, 100, 100))
        self.assertFalse(excluded(9, 100, 100))

    def test_r_boundary_and_otherwise_high_p(self):
        self.assertFalse(excluded(11, 100, 55))
        self.assertTrue(excluded(11, 100, 56))
        self.assertFalse(excluded(11, 100, 54))

    def test_unassigned_and_rounded_display_do_not_change_rule(self):
        self.assertFalse(excluded(0, 100, 0))
        self.assertTrue(excluded(10001, 100000, 100000))
        self.assertFalse(excluded(10000, 100000, 100000))
        with self.assertRaises(ValueError):
            excluded(11, 10, 100)


if __name__ == '__main__':
    unittest.main()
