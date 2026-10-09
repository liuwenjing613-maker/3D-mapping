import unittest
from continuity_policy import track_window, in_window, DirectionalGate


class ContinuityTests(unittest.TestCase):
    def test_one_gap_recovers_only_under_gap1(self):
        a = [5, 0, 8, 9, 0, 0, 12]
        self.assertEqual(track_window(a, 0, 0)['allowed_frame_range'], [0, 0])
        w = track_window(a, 0, 1)
        self.assertEqual(w['allowed_frame_range'], [0, 4])
        self.assertEqual(w['directions']['forward']['trigger_empty_frames'], [4, 5])
        self.assertFalse(in_window(w, 6))

    def test_reverse_has_its_own_seed_only_history(self):
        a = [8, 0, 0, 7, 9, 0, 6]
        w = track_window(a, 4, 1)
        self.assertEqual(w['allowed_frame_range'], [2, 6])
        self.assertEqual(w['directions']['reverse']['trigger_empty_frames'], [2, 1])
        self.assertIsNone(w['directions']['forward']['stop_frame'])

    def test_original_frames_not_mapping_stride(self):
        a = [9] * 11
        a[2] = a[3] = 0
        w = track_window(a, 0, 1)
        self.assertFalse(in_window(w, 5))
        self.assertFalse(in_window(w, 10))

    def test_tracks_do_not_stop_each_other(self):
        self.assertFalse(in_window(track_window([5, 0, 0, 8], 0, 1), 3))
        self.assertTrue(in_window(track_window([5, 6, 7, 8], 0, 1), 3))

    def test_seed_at_each_boundary_and_empty_tail(self):
        self.assertEqual(track_window([5], 0, 0)['allowed_frame_range'], [0, 0])
        self.assertEqual(track_window([0, 5], 1, 1)['allowed_frame_range'], [0, 1])
        self.assertEqual(track_window([5, 0], 0, 1)['retained_nonempty_frames'], 1)

    def test_gap1_contains_strict_window_and_matches_independent_oracle(self):
        import itertools
        for present in itertools.product((0, 1), repeat=7):
            if not present[3]:
                continue
            windows = [track_window(present, 3, gap) for gap in (0, 1)]
            self.assertTrue(all(not in_window(windows[0], f) or in_window(windows[1], f) for f in range(7)))
            for max_gap, window in enumerate(windows):
                expected = {3}
                for travel in ([4, 5, 6], [2, 1, 0]):
                    seen = []
                    for f in travel:
                        seen.append(present[f])
                        if len(seen) >= max_gap + 1 and not any(seen[-(max_gap + 1):]):
                            break
                        expected.add(f)
                self.assertEqual({f for f in range(7) if in_window(window, f)}, expected)

    def test_bad_seed_rejected(self):
        for areas, seed in [([], 0), ([0, 0], 1), ([2], 1), ([-1, 5], 1)]:
            with self.assertRaises(ValueError):
                track_window(areas, seed, 1)

    def test_streaming_gate_equals_saved_trajectory_windows(self):
        import itertools
        for present in itertools.product((0, 1), repeat=7):
            if not present[3]: continue
            for gap in (0, 1):
                window = track_window(present,3,gap)
                for step in (-1, 1):
                    gate = DirectionalGate([1],3,step,gap)
                    for fid in range(3+step,7 if step>0 else -1,step):
                        alive = gate.consume(fid,{1:present[fid]})
                        self.assertEqual(1 in alive,in_window(window,fid))

    def test_streaming_gate_does_not_restart_or_share_missing_counts(self):
        gate = DirectionalGate([1,2],0,1,1)
        self.assertEqual(gate.consume(1,{1:0,2:10}),[1,2])
        self.assertEqual(gate.consume(2,{1:0,2:10}),[2])
        self.assertEqual(gate.consume(3,{1:100,2:10}),[2])
        self.assertEqual(gate.stops,{1:2})

    def test_streaming_gate_rejects_stride_and_missing_target(self):
        gate = DirectionalGate([1,2],0,1,1)
        with self.assertRaises(ValueError): gate.consume(5,{1:5,2:5})
        with self.assertRaises(ValueError): gate.consume(1,{1:5})


if __name__ == '__main__':
    unittest.main()
