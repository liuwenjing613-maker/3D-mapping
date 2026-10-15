import unittest
import numpy as np
from mask_local_ops import accepted_owner, authorized, scoped_history, assert_source_scope


class MaskLocalTests(unittest.TestCase):
    def test_rejected_mask_has_no_permission(self):
        with self.assertRaises(ValueError):
            accepted_owner(np.array([[0, 1]]), [], {1: 1}, {1: 101})

    def test_empty_accepted_frame_cannot_reassociate_even_with_core(self):
        ids, changed = scoped_history(np.array([0, 1]), np.array([7, 8]),
            np.ones(2, bool), np.array([101, 101]), np.array([101, 101]),
            np.ones(2, bool), np.zeros(2, int))
        np.testing.assert_array_equal(ids, [7, 8])
        self.assertFalse(changed.any())

    def test_same_frame_different_target_not_authorized(self):
        np.testing.assert_array_equal(authorized(np.array([101, 102]), np.array([102, 101])), [False, False])

    def test_removed_overlap_and_outside_pixels_preserved(self):
        ids, _ = scoped_history(np.arange(3), np.array([7, 7, 7]), np.ones(3, bool),
            np.full(3, 101), np.full(3, 101), np.ones(3, bool), np.array([101, 0, 0]))
        np.testing.assert_array_equal(ids, [101, 7, 7])

    def test_retired_source_cannot_be_resurrected(self):
        ids, changed = scoped_history(np.array([0]), np.array([7]), np.array([False]),
            np.array([101]), np.array([101]), np.array([True]), np.array([101]))
        np.testing.assert_array_equal(ids, [7]); self.assertFalse(changed.any())

    def test_prior_and_expansion_both_require_scope(self):
        ids, _ = scoped_history(np.arange(4), np.full(4, 7), np.ones(4, bool),
            np.array([101, 101, -1, -1]), np.array([102, 102, 102, 102]),
            np.array([True, True, True, False]), np.array([101, 0, 102, 102]))
        np.testing.assert_array_equal(ids, [101, 7, 102, 7])

    def test_aliases_resolve_only_final_adopted_identity(self):
        a = accepted_owner(np.array([[0, 1]]), [1, 2], {1: 1, 2: 1}, {1: 101})
        np.testing.assert_array_equal(a, [[0, 101]])

    def test_source_pixel_not_shared_tsdf_index_is_permission(self):
        # Two pixels project to the same point; only one is in the adopted mask.
        ids, _ = scoped_history(np.array([0, 0]), np.array([7, 7]), np.ones(2, bool),
            np.array([101]), np.array([101]), np.array([True]), np.array([101, 0]))
        np.testing.assert_array_equal(ids, [101, 7])
        candidates = np.unique(ids)  # Old outside source remains a candidate.
        np.testing.assert_array_equal(candidates, [7, 101])

    def test_guard_rejects_outside_deletion(self):
        with self.assertRaises(AssertionError):
            assert_source_scope(np.array([7]), np.array([0]), np.array([0]))

    def test_guard_rejects_wrong_target_relabel(self):
        with self.assertRaises(AssertionError):
            assert_source_scope(np.array([7]), np.array([102]), np.array([101]))

    def test_allowed_deletion_and_relabel(self):
        np.testing.assert_array_equal(assert_source_scope(np.array([7, 7, 7]),
            np.array([0, 101, 7]), np.array([101, 101, 0])), [True, True, False])


if __name__ == '__main__':
    unittest.main()
