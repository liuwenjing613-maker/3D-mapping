"""Expose the fixed exclusion rule as exactly two groups, without rerunning inference."""
import argparse
import hashlib
import json
from pathlib import Path


def main(root):
    root=Path(root)
    data=json.loads((root/'comparison.json').read_text(encoding='utf-8'))
    original=hashlib.sha256((root/'classification.json').read_bytes()).hexdigest()
    assert original==data['classification_sha256']
    groups={'excluded':[],'retained':[]}
    for rec in data['records']:
        assert rec['exclude']==(rec['high_excluded_fraction']>=data['config']['mask_fraction'])
        rec['decision']='exclude' if rec['exclude'] else 'retain'
        groups['excluded' if rec['exclude'] else 'retained'].append({k:rec[k] for k in ['group_id','original_track_id','mask_id','frame','mask_sha256','decision','high_excluded_fraction']})
    assert sum(map(len,groups.values()))==len(data['records'])==len({r['group_id'] for r in data['records']})
    data['stats'].update(excluded=len(groups['excluded']),retained=len(groups['retained']))
    for key in ['suggest_exclude','retain','review','operationally_kept']:data['stats'].pop(key,None)
    data['config']['decision_rule']='exclude when high-confidence excluded pixels / unchanged mask area >= 0.9; otherwise retain. Exactly two groups.'
    data['schema']='auto-v1-seed-semantic-binary-comparison-v1'
    data['presentation_policy']='User requested two groups only; original exclusion decisions unchanged.'
    for key in ['audit','audit_note','sensitivity']:data.pop(key,None)
    selection={'schema':'auto-v1-semantic-binary-selection-v1','classification_sha256':original,
               'pixel_confidence':data['config']['pixel_confidence'],'mask_fraction':data['config']['mask_fraction'],
               'exclude_classes':data['config']['exclude_classes'],'counts':{k:len(v) for k,v in groups.items()},**groups}
    for name,value in [('binary_comparison.json',data),('selection_binary.json',selection)]:
        (root/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(selection['counts']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
