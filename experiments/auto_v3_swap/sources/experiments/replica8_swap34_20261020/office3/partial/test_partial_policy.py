import unittest
from copy import deepcopy
from partial_policy import partial_decision

class PartialPolicyTest(unittest.TestCase):
    def test_changes_only_whole_actions(self):
        original = {'frame': 5, 'accepted_track_ids': [3], 'objects': {'3': {'gate': True}},
                    'old_observation_plans': {'7': 'whole', '8': 'partial'}}
        snapshot = deepcopy(original)
        expected = deepcopy(original)
        expected['old_observation_plans']['7'] = 'partial'
        revised = partial_decision(original)
        self.assertEqual(revised, expected)
        self.assertEqual(original, snapshot)
        revised['objects']['3']['gate'] = False
        self.assertEqual(original, snapshot)

    def test_no_accepted_masks_keeps_original_decision(self):
        decision = {'frame': 10, 'accepted_track_ids': [], 'objects': {}}
        self.assertEqual(partial_decision(decision), decision)

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(AssertionError):
            partial_decision({'old_observation_plans': {'1': 'unknown'}})

if __name__ == '__main__':
    unittest.main()
