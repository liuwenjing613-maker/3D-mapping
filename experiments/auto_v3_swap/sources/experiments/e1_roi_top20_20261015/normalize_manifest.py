"""Make frozen point membership agree with the 32 displayed unique ROIs."""
import numpy as np
from prepare_inputs import ROOT,SOURCE,read,write,sha
p=SOURCE/'room2';manifest=read(p/'manifest.json');inputs=read(ROOT/'input_manifest.json')
if len(manifest['items'])==32:
 print('MANIFEST_ALREADY_UNIQUE',flush=True)
else:
 assert len(manifest['items'])==60
 byuid={};cores=[];contexts=[];items=[]
 with np.load(p/'membership.npz') as z:
  for i,item in enumerate(manifest['items']):
   uid=item['uid']
   if uid not in byuid:
    k=len(items);byuid[uid]=k
    cores.append(z['core_point_index'][z['core_offsets'][i]:z['core_offsets'][i+1]].copy())
    contexts.append(z['context_point_index'][z['context_offsets'][i]:z['context_offsets'][i+1]].copy())
    items.append(dict(item,fixed_order=k+1,entries=list(item['entries'])))
   else:
    k=byuid[uid];np.testing.assert_array_equal(cores[k],z['core_point_index'][z['core_offsets'][i]:z['core_offsets'][i+1]])
    items[k]['entries'].extend(item['entries'])
 for item,case in zip(items,inputs['cases']):
  assert item['uid']==case['uid']
  item['all_aliases']=case['aliases'];item['rankings']=case['rankings']
 np.savez_compressed(p/'membership.npz',core_offsets=np.r_[0,np.cumsum([len(x) for x in cores])],core_point_index=np.concatenate(cores).astype(np.int32),context_offsets=np.r_[0,np.cumsum([len(x) for x in contexts])],context_point_index=np.concatenate(contexts).astype(np.int32),uid=np.asarray([x['uid'] for x in items]))
 manifest.update(items=items,fixed_membership_sha256=sha(p/'membership.npz'))
 write(p/'manifest.json',manifest)
 summary=read(SOURCE/'selection_summary.json');olditems=summary['scenes'][0]['items'];merged=[]
 for case in inputs['cases']:
  sourceitems=[x for x in olditems if x['uid']==case['uid']];views=[]
  for si in sourceitems:
   for v in si['views']:views.append(dict(v,source_view=v['view'],source_entry=si['preferred_entry'],view=len(views)+1))
  assert len(views)==len(case['views'])
  merged.append(dict(sourceitems[0],fixed_order=case['order'],all_aliases=case['aliases'],rankings=case['rankings'],views=views))
 summary['scenes'][0]['items']=merged;write(SOURCE/'selection_summary.json',summary)
 hashes=read(SOURCE/'files_sha256.json')
 for name in ['room2/manifest.json','room2/membership.npz','selection_summary.json']:hashes[name]=sha(SOURCE/name)
 write(SOURCE/'files_sha256.json',hashes)
 inputs['source_summary_sha256']=sha(SOURCE/'selection_summary.json');write(ROOT/'input_manifest.json',inputs)
 print('MANIFEST_NORMALIZED',len(items),sum(len(x['entries']) for x in items),flush=True)
