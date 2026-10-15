from pathlib import Path
H=Path(__file__).resolve().parent
W=H.parents[1]
OLD=W/'outputs/strict_vote_annotation_20261010/room2'
C='/home/chenkejun/CVPR/experiments/e1_roi_top10_20261014'
R='/data/chenkejun/CVPR/results/e1_roi_top10_20261014'
for name in ['prepare_inputs.py','infer_cropformer.py','infer_sam.py','render_helpers.py','render_annotation.py']:
 t=(OLD/name).read_text(encoding='utf-8')
 t=t.replace('/home/chenkejun/CVPR/experiments/strict_vote_annotation_20261010/room2',C).replace('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/room2',R)
 t=t.replace('room2-strict-vote-top30-20261010','room2-e1-prerepair-CUT-top10-20261014')
 if name=='prepare_inputs.py':
  t=t.replace("OLD=Path('/data/chenkejun/CVPR/revisable_instance_map/roi_threeway_fixed_20261006/selection-20261006-f49eb927')","OLD=Path('/data/chenkejun/CVPR/results/strict_vote_annotation_20261010/room2')")
  t=t.replace("entries=sorted((e for e in catalog['entries'] if e['group']=='CONFLICT'),key=lambda e:e['rank'])", "entries=[e for e in catalog['entries'] if e['rank']<=10]")
  t=t.replace("assert len(entries)==30 and [e['id'] for e in entries]==[f'ROI-C{i:04d}' for i in range(1,31)]", "assert len(entries)==30 and all(sum(e['group']==g for e in entries)==10 for g in ['CONFLICT','U','T'])")
  t=t.replace("assert len({e['uid'] for e in entries})==30", "assert len({e['uid'] for e in entries})==19")
  t=t.replace("assert [e['count'] for e in entries]==sorted((e['count'] for e in entries),reverse=True)","assert all([e['count'] for e in entries if e['group']==g]==sorted([e['count'] for e in entries if e['group']==g],reverse=True) for g in ['CONFLICT','U','T'])")
  t=t.replace('unique_selected_ROIs=30','unique_selected_ROIs=19,ranked_entries=30')
  t=t.replace("selection_basis='strict-vote CONFLICT absolute point count descending; original component ID breaks ties'", "selection_basis='C/U/T each top10 absolute point count; component ID ties; 30 entries / 19 unique components'")
  t=t.replace("group='CONFLICT',views=[]", "group=e['group'],views=[]")
  t=t.replace("target=alarm[:,:,2]>0", "target=alarm[:,:,[0,1,2][{'U':0,'T':1,'CONFLICT':2}[e['group']]]]>0")
  t=t.replace("case['views'].append(view);", "view.update(trigger_group=e['group'],source_entry=e['id'],source_rank=e['rank'],source_view=v['view']);case['views'].append(view);")
  anchor=" first=cases[0]['views'][0];"
  merge=""" # Same physical component appears once; retain all three views from each qualifying state ranking.
 merged={}
 for case in cases:
  uid=case['uid']
  if uid not in merged:
   merged[uid]=dict(case,order=len(merged)+1,rank=len(merged)+1,views=[],rankings=[],aliases=[])
  dest=merged[uid];dest['rankings'].append(dict(id=case['ROI'],group=case['group'],rank=case['rank'],count=case['counts'][case['group']]))
  dest['aliases'].append(case['ROI'])
  for view in case['views']:
   view['number']=len(dest['views'])+1;dest['views'].append(view)
 cases=list(merged.values())
 assert len(cases)==19 and sum(len(c['views']) for c in cases)==90
"""
  t=t.replace(anchor,merge+anchor)
  t=t.replace("selected_ROIs=30,preferred_views=90", "selected_ROIs=19,ranked_entries=30,preferred_views=90")
  t=t.replace("ROI_count=30,items=summary", "ROI_count=19,ranked_entries=30,items=summary")
  t=t.replace('selected_cases=30','selected_cases=19,ranked_entries=30')
  t=t.replace("ranking_group='CONFLICT'", "ranking_group='CONFLICT/U/T'")
  t=t.replace('dict(cases=30,views=90','dict(cases=19,ranked_entries=30,views=90')
 if name=='render_annotation.py':
  t=t.replace('def main():','def main():\n import normalize_manifest')
  t=t.replace('for channel,color,code in [(0,(140,145,155),1),(1,(255,190,30),2),(2,(245,60,70),3)]:',"for channel,color,code in sorted([(0,(140,145,155),1),(1,(255,190,30),2),(2,(245,60,70),3)],key=lambda row:row[0]=={'U':0,'T':1,'CONFLICT':2}[v['trigger_group']]):")
  t=t.replace("scene_counts={'room2':30}","scene_counts={'room2':19}")
  t=t.replace("pr=dict(group='CONFLICT'", "pr=dict(group=v['trigger_group']")
  t=t.replace("len(viewer['cases'])==30", "len(viewer['cases'])==19")
  t=t.replace('case_count=30','case_count=19,ranked_entries=30')
  t=t.replace("ranking_group='CONFLICT',ranking='strict-vote absolute conflict point count descending; original component ID for ties'", "ranking_group='CONFLICT/U/T',ranking='C/U/T each top10; absolute point count; 30 entries / 19 unique components'")
  t=t.replace('views=90,crops=', 'views=90,crops=')
  t=t.replace('cvpr-room2-strict-vote-selection-20261010-v1','cvpr-room2-e1-prerepair-cut-top10-20261014-v1')
  t=t.replace('room2_strict_vote_人工标注_20261010.json','room2_E1修复前三状态前十_质量判定_20261014.json')
  t=t.replace('room2_top30_candidates.zip','room2_CUT_top10_candidates.zip')
  t=t.replace('dict(cases=30,views=90','dict(cases=19,ranked_entries=30,views=90')
 (H/name).write_text(t,encoding='utf-8',newline='\n')
 compile(t,str(H/name),'exec')
print('ASSEMBLED',C,R)
