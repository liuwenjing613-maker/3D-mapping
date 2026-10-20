"""Replace one frame's sparse votes once; input arrays are never modified."""
import hashlib
import numpy as np


class EvidenceReplacement:
    def __init__(self, keys, votes):
        self.keys = np.asarray(keys, np.int64).copy()
        self.votes = np.asarray(votes, np.int64).copy()
        if len(self.keys) != len(self.votes) or np.any(self.votes <= 0):
            raise ValueError('Invalid baseline vote table')
        if np.any(self.keys[1:] <= self.keys[:-1]):
            raise ValueError('Baseline keys must be sorted and unique')
        self.transactions = {}

    @staticmethod
    def signature(old, new):
        h = hashlib.sha256()
        for values in (old, new):
            a = np.asarray(values, '<i8')
            h.update(np.asarray([len(a)], '<i8').tobytes())
            h.update(a.tobytes())
        return h.hexdigest()

    @staticmethod
    def add_delta(keys, votes, delta_keys, sign):
        all_keys = np.concatenate([keys, delta_keys])
        all_votes = np.concatenate([votes, np.full(len(delta_keys), sign, np.int64)])
        order = np.argsort(all_keys, kind='stable')
        all_keys, all_votes = all_keys[order], all_votes[order]
        unique, starts = np.unique(all_keys, return_index=True)
        totals = np.add.reduceat(all_votes, starts)
        if np.any(totals < 0):
            raise ValueError('Cannot retract votes absent from the current ledger')
        keep = totals > 0
        return unique[keep], totals[keep]

    def replace(self, transaction_id, old_frame_keys, new_frame_keys):
        old = np.asarray(old_frame_keys, np.int64)
        new = np.asarray(new_frame_keys, np.int64)
        for a in (old, new):
            if np.any(a[1:] <= a[:-1]):
                raise ValueError('One frame must supply unique sorted point/instance votes')
        signature = self.signature(old, new)
        if transaction_id in self.transactions:
            if self.transactions[transaction_id]['signature'] != signature:
                raise ValueError('Transaction ID reused with different evidence')
            return False
        # Validate retraction before insertion, so missing old evidence cannot be
        # concealed by an otherwise cancelling addition.
        keys, votes = self.add_delta(self.keys, self.votes, old, -1)
        keys, votes = self.add_delta(keys, votes, new, 1)
        self.keys, self.votes = keys, votes
        self.transactions[transaction_id] = {'old': old.copy(), 'new': new.copy(), 'signature': signature}
        return True

    def undo(self, transaction_id):
        transaction = self.transactions[transaction_id]
        keys, votes = self.add_delta(self.keys, self.votes, transaction['new'], -1)
        keys, votes = self.add_delta(keys, votes, transaction['old'], 1)
        self.keys, self.votes = keys, votes
        del self.transactions[transaction_id]

    def digest(self):
        h = hashlib.sha256()
        h.update(self.keys.astype('<i8', copy=False).tobytes())
        h.update(self.votes.astype('<i8', copy=False).tobytes())
        return h.hexdigest()
