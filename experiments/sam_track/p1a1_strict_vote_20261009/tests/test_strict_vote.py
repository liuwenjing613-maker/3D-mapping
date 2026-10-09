"""Portable entry for the same ten measured semantic regression tests."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from strict_vote_ops import strict_frame_keys,revised_frame,frame_delta
from evidence_replacement import EvidenceReplacement


def reference(raw,base):
    bypoint={}
    for key in raw:bypoint.setdefault(int(key)//base,set()).add(int(key)%base)
    return np.asarray(sorted(p*base+next(iter(ids)) for p,ids in bypoint.items() if len(ids)==1),np.int64)


class StrictRepairTests(unittest.TestCase):
    def test_single_id_and_same_id_shared_support(self):
        np.testing.assert_array_equal(strict_frame_keys([10,10,110,120],100),[10])

    def test_shared_mask_retirement_keeps_effective_vote(self):
        before=strict_frame_keys([10,10,120],100)
        after,_=revised_frame([0,1],[10,20],[],[],100)
        np.testing.assert_array_equal(before,after)

    def test_old_and_new_must_be_reduced_together(self):
        after,raw=revised_frame([0],[10],[0],[20],100)
        np.testing.assert_array_equal(raw,[10,20]);self.assertEqual(len(after),0)
        self.assertEqual(len(np.union1d(strict_frame_keys([10],100),strict_frame_keys([20],100))),2)

    def test_removing_conflicting_mask_releases_retained_vote(self):
        before,after,removed,added=frame_delta([10,20],[10],100)
        self.assertEqual(len(before),0);np.testing.assert_array_equal(after,[10])
        self.assertEqual(len(removed),0);np.testing.assert_array_equal(added,[10])

    def test_adding_conflicting_mask_retracts_old_effective_vote(self):
        before,after,removed,added=frame_delta([10],[10,20],100)
        np.testing.assert_array_equal(before,[10]);self.assertEqual(len(after),0)
        np.testing.assert_array_equal(removed,[10]);self.assertEqual(len(added),0)

    def test_new_support_of_same_identity_is_one_vote(self):
        after,_=revised_frame([0,0],[10,10],[0,0],[10,10],100)
        np.testing.assert_array_equal(after,[10])

    def test_partial_retirement_preserves_same_mask_residual(self):
        after,_=revised_frame([0,1],[10,10],[2],[20],100)
        np.testing.assert_array_equal(after,[10,110,220])

    def test_independent_randomized_replay_and_exact_rollback(self):
        rng=np.random.default_rng(20261009)
        for trial in range(100):
            raw=[rng.integers(0,20,100)*100+rng.integers(1,7,100) for _ in range(5)]
            frames=[strict_frame_keys(x,100) for x in raw]
            for a,b in zip(raw,frames):np.testing.assert_array_equal(b,reference(a,100))
            k,v=np.unique(np.concatenate(frames),return_counts=True)
            ledger=EvidenceReplacement(k,v);original=ledger.digest()
            fid=trial%5;changed=raw[fid][rng.random(len(raw[fid]))>.4]
            changed=np.r_[changed,rng.integers(0,20,10)*100+rng.integers(1,7,10)]
            after=strict_frame_keys(changed,100)
            self.assertTrue(ledger.replace('frame',frames[fid],after))
            frames[fid]=reference(changed,100)
            nk,nv=np.unique(np.concatenate(frames),return_counts=True)
            np.testing.assert_array_equal(ledger.keys,nk);np.testing.assert_array_equal(ledger.votes,nv)
            digest=ledger.digest();self.assertFalse(ledger.replace('frame',strict_frame_keys(raw[fid],100),after))
            self.assertEqual(digest,ledger.digest());ledger.undo('frame');self.assertEqual(original,ledger.digest())

    def test_missing_vote_cannot_be_hidden_by_addition(self):
        ledger=EvidenceReplacement([10],[1]);digest=ledger.digest()
        with self.assertRaises(ValueError):ledger.replace('invalid',[20],[20])
        self.assertEqual(digest,ledger.digest())

    def test_order_invariance_and_empty_frames(self):
        raw=np.array([10,20,110,110,220]);rng=np.random.default_rng(9)
        for _ in range(10):np.testing.assert_array_equal(strict_frame_keys(rng.permutation(raw),100),[110,220])
        self.assertEqual(len(strict_frame_keys([],100)),0)

if __name__=='__main__':unittest.main(verbosity=2)
