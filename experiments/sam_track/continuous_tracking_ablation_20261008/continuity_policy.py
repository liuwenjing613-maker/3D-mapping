"""Stop each saved mask irreversibly after an excess gap, from its seed in each direction."""


def track_window(areas, seed_frame, max_empty_gap):
    if max_empty_gap not in (0, 1):
        raise ValueError('Only the two predeclared gap policies are supported')
    areas = [int(v) for v in areas]
    if not areas or any(v < 0 for v in areas):
        raise ValueError('Areas must be a complete sequence of nonnegative pixel counts')
    if not 0 <= seed_frame < len(areas) or areas[seed_frame] == 0:
        raise ValueError('The human seed must be nonempty and in the original sequence')
    bounds = [0, len(areas) - 1]
    directions = {}
    for name, step, bound_index in [('reverse', -1, 0), ('forward', 1, 1)]:
        gap = []
        last_present = seed_frame
        stop = None
        for fid in range(seed_frame + step, len(areas) if step > 0 else -1, step):
            if areas[fid] > 0:
                gap = []
                last_present = fid
            else:
                gap.append(fid)
                if len(gap) > max_empty_gap:
                    stop = fid
                    bounds[bound_index] = fid - step
                    break
        directions[name] = {
            'stop_frame': stop,
            'trigger_empty_frames': gap if stop is not None else [],
            'last_nonempty_frame_before_stop': last_present if stop is not None else None,
        }
    allowed = [bounds[0] <= fid <= bounds[1] for fid in range(len(areas))]
    assert allowed[seed_frame]
    return {
        'max_empty_gap': max_empty_gap,
        'seed_frame': seed_frame,
        'allowed_frame_range': bounds,
        'directions': directions,
        'raw_nonempty_frames': sum(v > 0 for v in areas),
        'retained_nonempty_frames': sum(v > 0 and keep for v, keep in zip(areas, allowed)),
        'removed_nonempty_frames': sum(v > 0 and not keep for v, keep in zip(areas, allowed)),
        'raw_mask_pixel_frames': sum(areas),
        'retained_mask_pixel_frames': sum(v for v, keep in zip(areas, allowed) if keep),
    }


def in_window(window, frame):
    a, b = window['allowed_frame_range']
    return a <= frame <= b


class DirectionalGate:
    """Streaming counterpart; already stopped targets cannot be revived by new nonempty masks."""

    def __init__(self, track_ids, seed_frame, step, max_empty_gap):
        if step not in (-1, 1) or max_empty_gap not in (0, 1):
            raise ValueError('Use one original-frame direction and a declared gap policy')
        self.ids = tuple(sorted(map(int, track_ids)))
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Track IDs must be distinct')
        self.last_frame = seed_frame
        self.step = step
        self.max_empty_gap = max_empty_gap
        self.streaks = dict.fromkeys(self.ids, 0)
        self.stops = {}

    def consume(self, frame, raw_areas):
        if frame != self.last_frame + self.step:
            raise ValueError('Observe every consecutive original frame, once, outward from the seed')
        if set(raw_areas) != set(self.ids) or any(v < 0 for v in raw_areas.values()):
            raise ValueError('Supply all original saved-mask areas before applying the stop policy')
        for oid in self.ids:
            if oid in self.stops:
                continue
            self.streaks[oid] = self.streaks[oid] + 1 if raw_areas[oid] == 0 else 0
            if self.streaks[oid] > self.max_empty_gap:
                self.stops[oid] = frame
        self.last_frame = frame
        return [oid for oid in self.ids if oid not in self.stops]
