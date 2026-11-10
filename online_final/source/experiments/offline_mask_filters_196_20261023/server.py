from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
from common import *
from appdata import metadata,rows_page,detail,image_bytes,scene_data
from score_logic import pair_page
import background_view
import covis_view
import contain_view
import dual95_view
import fragment_view
import id_view
import smallmask_view
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def send(self,b,mime='application/json',code=200):
  self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store' if mime.startswith(('application/json','text/')) else 'private,max-age=3600');self.end_headers();self.wfile.write(b)
 def do_GET(self):
  try:
   u=urlparse(self.path);q={k:v[0] for k,v in parse_qs(u.query).items()}
   if u.path.startswith('/ply-compare/'):
    root=Path('/data/chenkejun/CVPR/results/requested_fixedH_main_v2b_20261108/direct_post_sparse/viewer').resolve();name=u.path[len('/ply-compare/'):] or 'index.html';p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file():return self.send(b'Not found','text/plain',404)
    mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.json':'application/json','.bin':'application/octet-stream','.ply':'application/octet-stream'}.get(p.suffix,'application/octet-stream')
    return self.send(p.read_bytes(),mime)
   if u.path in ['/pr-p80/','/pr-p80/index.html','/pr-p80/app.js','/pr-p80/data.json','/pr-p80/audit.json']:
    name=u.path.rsplit('/',1)[-1] or 'index.html';p=Path('/data/chenkejun/CVPR/results/offline_pr_p80_20261023')/name
    return self.send(p.read_bytes(),{'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','json':'application/json'}[p.suffix[1:]])
   if u.path in ['/validity/','/validity/data.json','/validity/audit.json']:
    root=Path('/data/chenkejun/CVPR/results/offline_unassigned_validity_20261108');name=u.path.rsplit('/',1)[-1] or 'index.html';return self.send((root/name).read_bytes(),'text/html; charset=utf-8' if name.endswith('.html') else 'application/json')
   if u.path.startswith('/validity/images/'):
    root=Path('/data/chenkejun/CVPR/results/offline_unassigned_validity_20261108/images');p=(root/u.path.rsplit('/',1)[-1]).resolve();assert p.parent==root and p.suffix=='.jpg';return self.send(p.read_bytes(),'image/jpeg')
   if u.path.startswith('/sameview-image/'):
    root=Path('/data/chenkejun/CVPR/results/offline_sameview_fixedH_v3_20261108/images');p=(root/u.path.rsplit('/',1)[-1]).resolve();assert p.parent==root and p.suffix=='.jpg';return self.send(p.read_bytes(),'image/jpeg')
   if u.path=='/contain040-compare':return self.send(Path('/data/chenkejun/CVPR/results/offline_contain040_pipeline_20261108/compare.html').read_bytes(),'text/html; charset=utf-8')
   if u.path=='/fragments/':return self.send(Path('/data/chenkejun/CVPR/results/offline_fragments_fullinstance_u30_20261108/index.html').read_bytes(),'text/html; charset=utf-8')
   if u.path in ['/','/index.html','/app.js','/style.css']:
    name=u.path.lstrip('/') or 'index.html';mime={'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name];return self.send((OUT/name).read_bytes(),mime)
   if u.path=='/audit.json':return self.send((OUT/'audit.json').read_bytes())
   if u.path=='/api/kmedoids-audit':return self.send(Path('/data/chenkejun/CVPR/results/offline_kmedoids_learned_20261108/audit.json').read_bytes())
   if u.path=='/api/background-audit':return self.send((background_view.ROOT/'audit.json').read_bytes())
   if u.path=='/api/meta':
    data=metadata();data['current_screening']=background_view.metadata();data['covis']=covis_view.metadata();data['contain']=contain_view.metadata();data['dual95']=dual95_view.metadata();data['fragment']=fragment_view.metadata();data['idmatch']=id_view.metadata();data['smallmask']=smallmask_view.metadata();data['names']={k:v for key,val in data['names'].items() for k,v in ([(key,val),('fragment_filter','排除碎片'),('id_matching','ID匹配'),('smallmask','同视图面积筛选')] if key=='skip_repair_B' else [(key,val)])}
   elif u.path=='/api/smallmask':data=smallmask_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/idmatch':data=id_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/fragment':data=fragment_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/fragment-audit':return self.send((fragment_view.ROOT/'audit.json').read_bytes())
   elif u.path=='/api/dual95':data=dual95_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/dual95-detail':data=dual95_view.detail(q['scene'],q['id'])
   elif u.path=='/dual95-image':return self.send(dual95_view.image_bytes(q['scene'],q['id']),'image/jpeg')
   elif u.path=='/api/dual95-audit':return self.send((dual95_view.ROOT/'audit.json').read_bytes())
   elif u.path=='/api/contain':data=contain_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/contain-detail':data=contain_view.detail(q['scene'],q['id'])
   elif u.path=='/api/contain-audit':return self.send((contain_view.ROOT/'audit.json').read_bytes())
   elif u.path=='/api/covis':data=covis_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/covis-detail':data=covis_view.detail(q['scene'],q['id'])
   elif u.path=='/api/covis-audit':return self.send((covis_view.ROOT/'audit.json').read_bytes())
   elif u.path=='/api/background':data=background_view.rows(q['scene'],int(q.get('page',0)),q.get('q',''),q.get('status','all'))
   elif u.path=='/api/background-detail':data=background_view.detail(q['scene'],q['id'])
   elif u.path=='/background-image':return self.send(background_view.image_bytes(q['scene'],q['id'],q.get('kind','context')),'image/jpeg')
   elif u.path=='/api/rows':
    allowed=None;subset=q.get('subset','all')
    if q['phase']=='grouping' and subset!='all':
     assert q['scene'] in SCENES and subset in ('pairs','pair_one','pair_two')
     experiment=read(Path('/data/chenkejun/CVPR/results/offline_kmedoids_learned_20261108')/(q['scene']+'.json'))
     allowed={gid for gid,g in experiment['groups'].items() if g['before']==2 and (subset=='pairs' or g['after']==(1 if subset=='pair_one' else 2))}
    data=rows_page(q['scene'],q['phase'],int(q.get('page',0)),q.get('q',''),q.get('status','all'),allowed_groups=allowed)
   elif u.path=='/api/group-kmedoids':
    assert q['scene'] in SCENES
    data=read(Path('/data/chenkejun/CVPR/results/offline_kmedoids_learned_20261108')/(q['scene']+'.json'))
   elif u.path=='/api/group-dedup':
    assert q['scene'] in SCENES
    data=read(Path('/data/chenkejun/CVPR/results/offline_group_dedup_07_20261108')/(q['scene']+'.json'))
   elif u.path=='/api/detail':data=detail(q['scene'],q['id'],q.get('phase','retained'))
   elif u.path=='/api/pairs':
    data=pair_page(q['scene'],q['phase'],q['id'],int(q.get('page',0)));s=scene_data(q['scene']);data['masks']={g:v['primary'] for g,v in s['groups'].items()}
   elif u.path in ['/image','/mask']:
    return self.send(image_bytes(q['scene'],q['id'],q.get('full')=='1',u.path=='/mask'),'image/png' if u.path=='/mask' else 'image/jpeg')
   else:return self.send(b'Not found','text/plain',404)
   self.send(json.dumps(data,ensure_ascii=False,allow_nan=False).encode())
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:self.send(json.dumps(dict(error=str(e)),ensure_ascii=False).encode(),code=500)
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',19063),Handler).serve_forever()
