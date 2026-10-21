"""Replace all old candidates only at points with one unique new identity."""
import numpy as np


def replace_frame_points(old_keys, new_keys, base):
    old = np.unique(np.asarray(old_keys, dtype=np.int64))
    new = np.unique(np.asarray(new_keys, dtype=np.int64))
    points, inverse, count = np.unique(new // base, return_inverse=True, return_counts=True)
    committed = new[count[inverse] == 1]
    conflict = points[count > 1]
    covered = committed // base
    retained = old[~np.isin(old // base, covered)]
    return np.union1d(retained, committed), retained, committed, conflict
