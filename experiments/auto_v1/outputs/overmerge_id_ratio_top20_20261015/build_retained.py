"""Reuse the existing ROI review UI with rejected labels removed from every surface."""
from pathlib import Path
import copy,hashlib,json,re
import numpy as np
from PIL import Image,ImageDraw,ImageFont

H=Path(__file__).resolve().parent;ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
OLD=ROOT/'room2_E1_prerepair_CUT_top20_20261015';AUDIT=ROOT/'room2_E1_overmerge_audit_top20_20261015'
OUT=ROOT/'room2_E1_retained_ROI_top20_20261015';(OUT/'assets').mkdir(parents=True,exist_ok=True)
PREFIX='../'+OLD.name+'/'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(path):
 a=np.asarray(Image.open(path),np.uint16);return a[:,:,0]+256*a[:,:,1]
def projected(labels,codes,name):
 yy,xx=np.nonzero(codes=={'U':1,'T':2,'C':3}[name]);rows=[]
 for lid,n in zip(*np.unique(labels[yy,xx],return_counts=True)):
  keep=labels[yy,xx]==lid;x,y=xx[keep],yy[keep];k=int(((x-x.mean())**2+(y-y.mean())**2).argmin())
  rows.append(dict(id=int(lid),count=int(n),x=int(x[k]),y=int(y[k]),projection_box=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]))
 return sorted(rows,key=lambda r:(-r['count'],r['id']))

def main():
 old_hashes={n:sha(OLD/n) for n in ['index.html','viewer_data.json','projection_annotations.json','selection_setup.json']}
 d=read(OLD/'viewer_data.json');p=read(OLD/'projection_annotations.json');a=read(AUDIT/'data.json');setup=read(OLD/'selection_setup.json')
 jobs={j['key']:j for j in a['jobs']};removed_occ=changed=0;changed_keys=[];removed_by_model=[0,0,0];counts=[]
 crop_views={key:v for c in d['cases'] for v in c['views'] for key in v['variants']}
 for key,crop in d['crops'].items():
  job=jobs[key];model_counts=[];changed_this=False;codes=None
  crop['RGB']=PREFIX+crop['RGB']
  for mi,model in enumerate(crop['models']):
   source=a['sources'][job['sources'][mi]];members=set(job['members'][mi]);rejected=sorted(int(lid) for lid,cid in source['mapping'].items() if cid in members and a['candidates'][cid]['decision']=='reject')
   model['removed_ids']=rejected;model['count_before_filter']=model['count']
   if not rejected:
    for field in ['color','boundary','ids_file']:model[field]=PREFIX+model[field]
    model['ids']=model['ids_file'];model_counts.append(model['count']);continue
   changed+=1;changed_this=True;removed_occ+=len(rejected);removed_by_model[mi]+=len(rejected)
   old_labels=decode(OLD/model['ids_file']);positive=set(map(int,np.unique(old_labels)))-{0}
   assert set(rejected)<=positive,(key,mi,rejected)
   remove=np.isin(old_labels,rejected);labels=old_labels.copy();labels[remove]=0
   name=key+'_'+['original','local','sam'][mi]
   for field in ['color','boundary']:
    pixels=np.asarray(Image.open(OLD/model[field]).convert('RGBA')).copy();pixels[remove]=0
    rel='assets/'+name+'_'+field+'.png';Image.fromarray(pixels).save(OUT/rel);model[field]=rel
   enc=np.zeros((*labels.shape,3),np.uint8);enc[:,:,0]=labels%256;enc[:,:,1]=labels//256
   model['ids_file']=model['ids']='assets/'+name+'_ids.png';Image.fromarray(enc).save(OUT/model['ids_file'])
   model['positions']=[r for r in model['positions'] if r['id'] not in rejected]
   model['colors']={lid:col for lid,col in model['colors'].items() if int(lid) not in rejected}
   model['count']=len(positive)-len(rejected);assert model['count']==len(model['colors'])
   model_counts.append(model['count'])
   if codes is None:
    alarm=np.asarray(Image.open(OLD/crop_views[key]['alarm']).convert('RGBA'));code=np.zeros(alarm.shape[:2],np.uint8);valid=alarm[:,:,3]>0
    code[valid]=1;code[valid&(alarm[:,:,0]>200)&(alarm[:,:,1]>120)]=2;code[valid&(alarm[:,:,0]>200)&(alarm[:,:,1]<120)]=3
    x0,y0,x1,y1=crop['box'];codes=code[y0:y1,x0:x1]
   prev=p['crops'][key]['models'][mi]
   nextrows={state:projected(labels,codes,state) for state in ['U','T','C']}
   for state,rows in nextrows.items():
    assert sum(r['count'] for r in rows)==p['crops'][key]['counts'][state]
    for row in prev[state]:
     if row['id'] and row['id'] not in rejected:assert next(x for x in rows if x['id']==row['id'])==row
    oldzero=sum(r['count'] for r in prev[state] if r['id']==0 or r['id'] in rejected)
    assert sum(r['count'] for r in rows if r['id']==0)==oldzero
   p['crops'][key]['models'][mi]=nextrows
  counts.append(dict(key=key,retained=model_counts,removed=[len(m['removed_ids']) for m in crop['models']]))
  if changed_this:changed_keys.append(key)
  if 'comparison' in crop:
   if not changed_this:crop['comparison']=PREFIX+crop['comparison']
   else:
    rgb=Image.open(OUT/crop['RGB']).convert('RGB');w=600;hh=round(rgb.height*w/rgb.width)
    card=Image.new('RGB',(w*4+60,hh+100),'#f7f9fc');dr=ImageDraw.Draw(card)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    dr.text((12,10),f'room2 / {job["ROI"]} / 帧 {job["frame"]} / 已排除拟过合并 Mask',font=font,fill='#23394b')
    images=[rgb]
    for mo in crop['models']:
     color=np.asarray(Image.open(OUT/mo['color']).convert('RGBA'));edge=np.asarray(Image.open(OUT/mo['boundary']).convert('RGBA'));base=np.asarray(rgb).copy();used=color[:,:,3]>0
     base[used]=(base[used]*.48+color[:,:,:3][used]*.52).astype(np.uint8);base[edge[:,:,3]>0]=[250,250,250];images.append(Image.fromarray(base))
    for i,(title,img) in enumerate(zip(['RGB','原图 CropFormer','局部 CropFormer','SAM2.1'],images)):
     x=12+i*(w+12);title+=f' · 保留 {model_counts[i-1]}' if i else '';dr.text((x,45),title,font=font,fill='#23394b');card.paste(img.resize((w,hh),Image.Resampling.LANCZOS),(x,80))
    crop['comparison']='assets/'+key+'_retained_comparison.jpg';card.save(OUT/crop['comparison'],quality=92)
 for c in d['cases']:
  for v in c['views']:
   v['full_rgb']=PREFIX+v['full_rgb'];v['alarm']=PREFIX+v['alarm']
 for pv in p['views'].values():pv['alarm_asset']=PREFIX+pv['alarm_asset']
 d['name']='room2-e1-retained-roi-top20-20261015'
 d['overmerge_filter']=dict(rule=a['rule'],unique_rejected=a['stats']['reject'],removed_occurrences=removed_occ,source_sha256=sha(AUDIT/'data.json'),audit_page='index.html')
 save(OUT/'viewer_data.json',d);save(OUT/'projection_annotations.json',p);save(OUT/'mask_counts.json',counts)
 html=(OLD/'index.html').read_text(encoding='utf-8')
 for tag,value in [('reviewData',d),('projectionData',p)]:
  encoded=json.dumps(value,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
  html,n=re.subn(r'(<script id="'+tag+r'" type="application/json">).*?(</script>)',lambda m:m[1]+encoded+m[2],html,flags=re.S);assert n==1
 html=html.replace('room2 · E1 修复前种子质量审核','room2 · 筛选后 ROI 质量检查')
 html=html.replace('冲突 / 未分配 / 证据不足各前 20 · 60 个名额去重为 32 个 ROI · 三路分割','三类各前 20 · 60 个入榜名额 / 32 个 ROI · 拟剔除 '+str(a['stats']['reject'])+' 个候选')
 html=re.sub(r'<div class="room2Workflow">.*?</div>', '<div class="room2Workflow">已按本轮规则排除拟过合并 Mask。仍按 ROI 切换视角、比较三种分割，点选合格 Mask 并保存；没有合格候选就选“都不可靠 / 不通过”。</div>',html,count=1)
 html=html.replace('cvpr-room2-e1-prerepair-cut-top20-20261015-v1','cvpr-room2-e1-retained-roi-top20-20261015-v1')
 html=html.replace("source_selection:'room2-e1-prerepair-CUT-top20-20261015'","source_selection:'room2-e1-retained-roi-top20-20261015',overmerge_filter:D.overmerge_filter")
 html=html.replace('room2_E1修复前三状态前二十_质量判定_20261015.json','room2_E1各前20筛选后ROI_质量判定_20261015.json')
 old_label="label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png'"
 html=html.replace(old_label,old_label+",review_label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter")
 html=html.replace("j.models[0].count+' 个可见 mask · 读取原整帧缓存后裁剪'","j.models[0].count+' 个保留 mask · 已排除 '+j.models[0].removed_ids.length+' 个'")
 html=html.replace("j.models[1].count+' 个可见 mask · 相同权重 / 0.5 阈值'","j.models[1].count+' 个保留 mask · 已排除 '+j.models[1].removed_ids.length+' 个'")
 html=html.replace("j.models[2].count+' 个可见 mask · Hiera Large / 官方 AMG 默认'","j.models[2].count+' 个保留 mask · 已排除 '+j.models[2].removed_ids.length+' 个'")
 html=html.replace('未落入 mask','未落入保留 mask')
 # Redirects from the audit page may restore its ROI / exact crop without selecting any mask.
 html=html.replace("chooseCase(filtered[0]);\n})();", "const requested=new URLSearchParams(location.search),requestedCase=D.cases.findIndex(c=>c.uid===requested.get('roi'));\nchooseCase(requestedCase>=0?requestedCase:filtered[0]);\nconst requestedCrop=requested.get('crop');if(requestedCrop){const ci=D.cases[activeCase],vi=ci.views.findIndex(v=>v.variants.includes(requestedCrop));if(vi>=0){el('view').value=String(vi);chooseView(vi,requestedCrop);}}\n})();")
 html=html.replace('label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter',"label_asset:chosenModel<0?null:'assets/'+activeKey+'_'+['original','local','sam'][chosenModel]+'_ids.png',review_label_asset:chosenModel<0?null:j.models[chosenModel].ids_file,overmerge_filter:D.overmerge_filter")
 (OUT/'index.html').write_text(html,encoding='utf-8')
 runtime=re.findall(r'<script>\s*(.*?)</script>',html,re.S);assert len(runtime)==1;(H/'retained_runtime.js').write_text(runtime[0],encoding='utf-8')
 setup.update(name=d['name'],status='READY_FOR_RETAINED_ROI_REVIEW',overmerge_filter=d['overmerge_filter'],isolated_storage_key='cvpr-room2-e1-retained-roi-top20-20261015-v1',export_filename='room2_E1各前20筛选后ROI_质量判定_20261015.json',human_records_created=0)
 save(OUT/'selection_setup.json',setup)
 report=dict(status='PASS',unique_rejected=a['stats']['reject'],removed_occurrences=removed_occ,removed_by_model=removed_by_model,changed_model_crops=changed,
  changed_crops=len(changed_keys),ROIs=32,view_entries=180,crops=len(d['crops']),remaining_occurrences=sum(sum(c['retained']) for c in counts),
  projection_counts_rebuilt=True,retained_IDs_and_pixels_unchanged=True,original_page_unchanged=True,storage_isolated=True)
 assert report['remaining_occurrences']+removed_occ==a['stats']['occurrences']
 assert removed_occ==sum(len(c['occ']) for c in a['candidates'] if c['decision']=='reject')
 assert old_hashes=={n:sha(OLD/n) for n in old_hashes}
 save(OUT/'projection_annotation_audit.json',report);save(H/'retained_build_audit.json',report)
 (OUT/'plan.txt').write_text('沿用原 ROI 三模型质量检查页面；仅清除本轮拟剔除的标签，不重新分割或重分配像素。阈值 n2/N >=20% 且 n2>=10。保留不等于质量通过。原始页面、原始标注和地图保持不变。人工记录保存在独立浏览器键，请导出 JSON。',encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
