"""Verify the downloaded scientific artifact and every local presentation asset."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,re
from PIL import Image
HERE=Path(__file__).resolve().parent
WEB=HERE.parents[1]/'results/固定案例_三模型对比_20261006/room2_top30_repair_20261008'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attributes):
        for key,value in attributes:
            if key in ['href','src'] and value:self.links.append(value)

def main():
    data=read(WEB/'comparison_data.json');record=read(HERE/'review_bundle.json');build=read(WEB/'build_receipt.json')
    assert sha(HERE/'review_bundle.tar.gz')==record['sha256']==build['source_bundle_sha256']
    assert build['html_sha256']==sha(WEB/'index.html')
    assert build['code_sha256']==sha(HERE/'build_review.py')
    assert sha(WEB/'human_selection.json')==data['annotation_sha256']==read(HERE/'task_paths.json')['annotation_sha256']
    assert len(data['human_selections'])==len(data['model']['cases'])==18
    assert len(data['seed_audits'])==25 and len(data['global_association']['objects'])==23
    assert data['evaluation']['v3_code_flags_protocol_unchanged'] and not data['evaluation']['GT_used_for_repair']
    cross=read(WEB/'cross_seed_review/comparison.json')
    assert cross['status']=='PASS' and cross['GT_used'] is False and cross['map_or_tracking_changed'] is False
    for cross_record in cross['comparisons']:
        assert sha(WEB/'cross_seed_review'/cross_record['asset'])==cross_record['asset_sha256']
    palette=read(WEB/'instance_colors.json')['id_colors']
    assert all(str(row['persistent_id']) in palette for row in data['global_association']['objects'])
    for entry in [data['model']['full'],*data['model']['cases'].values()]:
        p=WEB/entry['path'];assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
        with p.open('rb') as stream:header=stream.read(4096)
        offset=header.index(b'end_header\n')+len(b'end_header\n')
        points=int(re.search(rb'element vertex (\d+)',header).group(1))
        assert points==entry['points'] and offset+points*31==p.stat().st_size
    images={asset for row in data['views'] for asset in row['assets'].values()}|{row['asset'] for row in data['track_views']}
    for name in images:
        with Image.open(WEB/name) as image:image.verify()
    parser=Links();parser.feed((WEB/'index.html').read_text(encoding='utf-8'))
    for value in parser.links:
        assert (WEB/value).resolve().is_relative_to(WEB.resolve()) and (WEB/value).is_file(),value
    receipt={'status':'PASS','cases':18,'human_masks':25,'unique_track_identities':23,'PLY_files':19,
      'full_surface_points':data['model']['full']['points'],'images_verified':len(images),
      'html_bytes_match_receipt':True,'all_referenced_assets_exist':True,'annotation_sha256_exact':True,
      'review_bundle_sha256':record['sha256']}
    with (WEB/'local_validation.json').open('w',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))

if __name__=='__main__':main()
