// Only an explicit checkbox change hides an instance. Solo selection is separate.
export function hiddenInstanceIDs(state, follow) {
  if (!follow || !state?.c) return [];
  const objects = state.c.objects;
  const enabled = new Set(state.enabled_track_ids ?? objects.map(o => o.track_id));
  const activeIDs = new Set(objects.filter(o => enabled.has(o.track_id)).map(o => o.audit.persistent_id));
  return [...new Set(objects.filter(o => !enabled.has(o.track_id) && !activeIDs.has(o.audit.persistent_id)).map(o => o.audit.persistent_id))];
}
