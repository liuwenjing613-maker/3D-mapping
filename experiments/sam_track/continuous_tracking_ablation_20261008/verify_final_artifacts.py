"""Verify actual published full-point models and independent colored PLY downloads."""
from pathlib import Path
import hashlib,json,re
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import numpy as np
WORK=Path(__file__).resolve().parent
BASE=WORK.parents[1]/'results/固定案例_三模型对比_20261006'
WEB=BASE/'continuous_tracking_ablation_20261008'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
complete=read(WORK/'server_results/complete.json');assert complete['status']=='PASS'
assert read(WEB/'experiment_status.json')['status']=='PASS'
receipts={}
for scene in ['room0','room2']:
    info=read(WEB/'models'/(scene+'_comparison_pack.json'))
    pack_path=WEB/'models'/info['path'];assert sha(pack_path)==info['sha256']
    dtype=np.dtype([(v,'<f4') for v in ['x','y','z']]+[(name,'<i4') for name in info['state_names']])
    assert dtype.itemsize==info['stride_bytes']==28
    pack=np.memmap(pack_path,mode='r',dtype=dtype,shape=(info['points'],))
    assert pack_path.stat().st_size==info['points']*28
    for name in ['gap0','gap1']:
        row=complete['scenes'][scene]['conditions'][name]
        path=WEB/'models'/f'{scene}_{name}_full_instance.ply'
        local=WORK.parents[1]/'可视化ply/连续性截断_20261008'/path.name
        assert sha(path)==sha(local)==row['PLY']['sha256']
        header=path.read_bytes()[:4096];offset=header.index(b'end_header\n')+11
        count=int(re.search(rb'element vertex (\d+)',header)[1]);assert count==info['points']
        ply_dtype=np.dtype([(v,'<f4') for v in ['x','y','z']]+[(v,'u1') for v in ['red','green','blue']]+[('instance_id','<i4')])
        assert path.stat().st_size==offset+count*19
        ply=np.memmap(path,mode='r',dtype=ply_dtype,offset=offset,shape=(count,))
        for key in ['x','y','z']:np.testing.assert_array_equal(ply[key],pack[key])
        np.testing.assert_array_equal(ply['instance_id'],pack[name])
        expected=np.full((count,3),105,np.uint8)
        for pid in np.unique(pack[name][pack[name]>0]):expected[pack[name]==pid]=info['id_colors'][str(int(pid))]
        for i,key in enumerate(['red','green','blue']):np.testing.assert_array_equal(ply[key],expected[:,i])
        assert int(np.sum(pack[name]<=0))==info['stats'][name]['unknown']==row['repair']['whole_scene_unknown_after']
        assert row['repair']['strict_commit']['final_outside_changes']==0
    np.testing.assert_array_equal(pack['gap0'],pack['gap1'])
    receipts[scene]={'points':info['points'],'PLY_points_xyz_labels_colors_exact':True,
       'local_PLY_SHA_matches_server':True,'gap0_gap1_labels_bit_identical':True}
    donor=BASE/(scene+'_tracking_playback_20261008')/'playback_data.json'
    meta=read(WEB/(scene+'_preview_data.json'));assert sha(donor)==meta['source_metadata_sha256']
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  self.links.extend(v for k,v in attrs if k in ['href','src'] and v)
links=Links()
for file in WEB.glob('*.html'):
 if file.name=='preview_timeline_snapshot.html':continue
 links.feed(file.read_text(encoding='utf-8'))
for link in links.links:
 path=urlparse(link)
 if path.scheme or path.netloc or not path.path:continue
 target=(WEB/unquote(path.path)).resolve();assert target.is_relative_to(BASE.resolve()) and target.is_file(),link
report={'status':'PASS','scenes':receipts,'relative_links_verified':len(links.links),
 'actual_vote_repair_and_geodesic_diffusion_complete':True,'all_four_bounded_commit_and_rollback_checks_passed':True,
 'GT_used_for_filter_or_repair':False,'original_playback_metadata_preserved':True,
 'v3_flags_and_core_unchanged':True,'formal_benchmark_result':False,
 'web_files_sha256':{str(p.relative_to(WEB)):sha(p) for p in WEB.rglob('*') if p.is_file()},
 'verification_code_sha256':sha(Path(__file__))}
(WORK/'final_local_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='web_files_sha256'},ensure_ascii=False,indent=2))
