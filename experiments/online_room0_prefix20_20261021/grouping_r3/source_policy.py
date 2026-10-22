"""Window/cabinet SAM routing takes precedence over grouping and seed ranking."""

def replacement_ledger(crops):
    ledger = {}
    for crop_key, crop in crops.items():
        nodes = {n['key']: n for n in crop['nodes']}
        for n in crop['nodes']:
            row = crop['models'][n['model']]['labels'][str(n['label'])]
            if row.get('reason') != 'SEMANTIC_CROPFORMER_REPLACED':
                continue
            assert n['model'] in (0, 1)
            matches = [nodes[key] for key in row['semantic_matches']]
            assert matches and all(m['model'] == 2 and m.get('semantic_hit') for m in matches)
            families = sorted({m['semantic']['target_family'] for m in matches})
            assert all(f in ('window', 'cabinet') for f in families)
            ledger.setdefault(n['candidate_id'], []).append(dict(
                crop_key=crop_key, ROI=crop['ROI'], frame=crop['frame'], model=n['model'], label=n['label'],
                families=families, sam_candidate_ids=[m['candidate_id'] for m in matches]))
    return ledger

def eligible_aliases(aliases, ledger):
    # A duplicate occurrence in a different ROI cannot undo a SAM replacement.
    # If full-mask duplicates have different sources, keep only their SAM aliases.
    if any(a['model'] != 2 and a['candidate_id'] in ledger for a in aliases):
        return [a for a in aliases if a['model'] == 2]
    return list(aliases)

def assert_sam_only_groups(groups, seeds):
    for members in groups:
        if any(a['semantic_hit'] for i in members for a in seeds[i]['aliases']):
            assert all(a['model'] == 2 for i in members for a in seeds[i]['aliases']), \
                'Window/cabinet group contains a forbidden CropFormer candidate; do not export.'
