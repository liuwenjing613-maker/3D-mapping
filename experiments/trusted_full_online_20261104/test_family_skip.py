"""Exercise actual preparation selection block and actual empty queue main."""
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import ast,collections,copy,hashlib,json,os,textwrap,time
import numpy as np
C=Path(__file__).resolve().parent;SCENES=['room0','room1','room2','office0','office1','office2','office3','office4'];report={}
for scene in SCENES:
 code=(C/'online'/scene/'track_prepare.py').read_text()
 start=code.index(' masks={};batches={}')
 stop=code.index(' for fid,slots in sorted(batches.items()):',start)
 block=compile(textwrap.dedent(code[start:stop]),str(C/'online'/scene/'track_prepare.py'),'exec')
 originals={};images={};hashes={}
 S=Path('/synthetic/seeds');R=Path('/synthetic/tracking')
 for tid in [1,2,3]:
  frame=245+tid*5;mask=np.zeros((680,1200),np.uint8);mask[100:110,tid*30:tid*30+10]=255;path=S/f'{tid}.png';images[str(path)]=mask;hashes[str(path)]=str(tid)*64
  p={'id':f'M{tid:04d}','mask':f'{tid}.png','mask_sha256':hashes[str(path)],'touches_any_edge':False,'noise_decision':{'exclude':False},'representative':{'touches_crop':False,'model':0},'aliases':[],'frame':frame,'area':100}
  originals[tid]={'id':f'A{tid:03d}','primary':p}
 def run(have):
  saved={}
  def layers(fid,frame):
   a=np.zeros((680,1200),np.uint16);tid=(fid-245)//5
   if tid in have:a[100,tid*30]=tid
   return [a]
  gl={'np':np,'groups':copy.deepcopy(originals),'selected':copy.deepcopy(originals),'FIDS':tuple(range(250,500,5)),'GROUP':'B','OLD_COUNT':50,'S':S,'R':R,'sourcehash':{},'digest':lambda p:hashes[str(p)],'Image':SimpleNamespace(open=lambda p:images[str(p)]),'scene':SimpleNamespace(source=SimpleNamespace(load_frame=lambda fid:fid),identity_layers=layers),'f':SimpleNamespace(atomic_json=lambda p,d:saved.update({str(p):d}))}
  exec(block,gl)
  assert set(gl['masks'])==set(have)
  assert gl['N']==len(have)
  assert {x['track_id'] for x in gl['skipped_no_family']}=={1,2,3}-set(have)
  assert sum(len(batch) for batches in gl['batches'].values() for batch in batches)==len(have)
  audit=saved[str(R/'skipped_no_current_family.json')];assert audit['skipped_count']==3-len(have) and audit['retained_count']==len(have)
  return gl,audit
 full,a=run({1,2,3});assert full['skipped_no_family']==[]
 for tid in [1,2,3]:np.testing.assert_array_equal(full['masks'][tid],images[str(S/f'{tid}.png')]>0)
 partial,b=run({1,3});empty,d=run(set());assert empty['batches']=={} and empty['seedmap']==[] and empty['diagnostics']==[]
 # The exact queue main must complete zero cases without worker/GPU inference.
 source=(C/'online'/scene/'track_queue.py').read_text();tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
 saved={};ROOT=Path('/synthetic/root')
 records={str(ROOT/'inputs_complete.json'):{'sam_source_files':{}},str(R/'input_freeze.json'):{'seed_and_config_hashes':{}},str(R/'seed_diagnostics.json'):{'cases':[]}}
 gl={'Path':Path,'time':time,'ROOT':ROOT,'R':R,'read':lambda p:records[str(p)],'digest':lambda p:None,'dump':lambda p,v:saved.update({str(p):v}),'CHECKPOINT':100,'GROUP':'B','CURRENT_RAW':495,'RAW_COUNT':496,'ROWS':[],'ThreadPoolExecutor':ThreadPoolExecutor,'os':os,'worker':lambda c:(_ for _ in ()).throw(AssertionError('empty queue called worker'))}
 exec(compile(module,str(C/'online'/scene/'track_queue.py'),'exec'),gl);gl['main']()
 q=saved[str(R/'tracking_queue_complete.json')];assert q['status']=='PASS' and q['total_cases']==0 and q['cases']==[]
 assert "assert counts,('NO_CURRENT_OBSERVATION_FAMILY',tid,p['id'])" in code
 assert "seedhash[str(R / 'skipped_no_current_family.json')]" in code
 report[scene]={'all_have_family_unchanged':True,'partial_only_missing_skipped':True,'all_missing_empty_prepare':True,'actual_empty_queue_main_pass':True,'downstream_counts_assert_preserved':True}
out=Path('/data/chenkejun/CVPR/results/trusted_full_online_20261104/policy_tests/family_skip_compatibility.json');out.write_text(json.dumps({'status':'PASS','scenes':report,'scope':'actual selection block plus actual empty queue main; does not claim empty repair/commit support'},indent=2)+'\n');print('PASS all 8 scenes: all/partial/no family + actual empty queue')
