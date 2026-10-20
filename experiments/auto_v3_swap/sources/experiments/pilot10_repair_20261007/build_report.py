"""Build a local review page without changing original viewer or annotations."""
from pathlib import Path
import hashlib, json, tarfile
import shutil
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent.parent / 'results/固定案例_三模型对比_20261006'
OUT = DELIVERY / 'repair_validation_20261007'
bundle = HERE / 'review_bundle.tar.gz'
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == 'c5ae1737ee02c3be0e288bda7adc5cefb422aedd58e95103f216b31725e29420'
OUT.mkdir(exist_ok=True)
with tarfile.open(bundle, 'r:gz') as archive:
    for member in archive.getmembers():
        target = (OUT / member.name).resolve()
        if not target.is_relative_to(OUT.resolve()) or not (member.isfile() or member.isdir()):
            raise ValueError('Unsafe archive member')
    archive.extractall(OUT, filter='data')
hashes = json.loads((OUT / 'asset_hashes.json').read_text())
for path, expected in hashes.items():
    assert hashlib.sha256((OUT / path).read_bytes()).hexdigest() == expected
review = json.loads((OUT / 'review_data.json').read_text())
summary = json.loads((HERE / 'server_reports/v3_summary.json').read_text())
id_maps = json.loads((OUT / 'id_maps.json').read_text())
flags = []
for case, evaluated in zip(review['cases'], summary['cases']):
    for row in evaluated.get('target_source', []):
        hits = sorted((g for g in row['GT_overlap'] if g['included']), key=lambda g: -g['fraction'])
        if len(hits) < 2 or hits[1]['fraction'] < .2:
            continue
        view = case['views'][0]
        rgb = np.array(Image.open(OUT / view['assets']['RGB']).convert('RGB'))
        seed = np.array(Image.open(HERE / 'human_seeds_20ad6139511b' / (case['case_uid'] + '_seed.png')))
        x0, y0, x1, y1 = view['box']
        mask = seed[y0:y1, x0:x1] == row['track_id']
        image = np.rint(rgb.astype(float) * .62).astype(np.uint8)
        color = np.array(case['object_colors'][row['track_id']], np.uint8)
        image[mask] = np.rint(rgb[mask].astype(float) * .38 + color * .62).astype(np.uint8)
        filename = case['case_uid'] + '_mask' + str(row['native_mask_id']) + '_review.jpg'
        Image.fromarray(image).save(OUT / 'assets' / filename, quality=92)
        flags.append({'case_uid': case['case_uid'], 'scene': case['scene'], 'ROI': case['ROI'],
                      'native_mask_id': row['native_mask_id'], 'track_id': row['track_id'],
                      'major_GT_fractions': [g['fraction'] for g in hits[:2]],
                      'image': 'assets/' + filename})
assert review['status'] == summary['status'] == id_maps['status'] == 'PASS'
for case in review['cases']:
    for view in case['views']:
        original_alarm = DELIVERY / view['alarm_asset']
        alarm_copy = OUT / view['alarm_asset']
        shutil.copyfile(original_alarm, alarm_copy)
        assert hashlib.sha256(alarm_copy.read_bytes()).digest() == hashlib.sha256(original_alarm.read_bytes()).digest()
    view = case['views'][0]
    key = view['crop_key']
    rgb = np.array(Image.open(DELIVERY / 'assets' / (key + '_rgb.webp')).convert('RGB'))
    case['candidate_images'] = {}
    for model in ['original', 'local', 'sam']:
        label = np.array(Image.open(DELIVERY / 'assets' / (key + '_' + model + '_ids.png')))
        if label.ndim == 3:
            values = label.astype(np.uint32)
            label = values[:, :, 0] | (values[:, :, 1] << 8) | (values[:, :, 2] << 16)
        color = np.array(Image.open(DELIVERY / 'assets' / (key + '_' + model + '_color.png')).convert('RGB'))
        assert color.shape == rgb.shape and label.shape == rgb.shape[:2]
        candidate = rgb.copy()
        foreground = label > 0
        candidate[foreground] = np.rint(rgb[foreground].astype(float) * .45 + color[foreground] * .55).astype(np.uint8)
        filename = case['case_uid'] + '_candidate_' + model + '.jpg'
        Image.fromarray(candidate).save(OUT / 'assets' / filename, quality=92)
        case['candidate_images'][model] = 'assets/' + filename
data = json.dumps({'review': review, 'summary': summary, 'id_maps': id_maps, 'seed_review_flags': flags}, ensure_ascii=False).replace('<', '\\u003c')
html = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>10 例修复验证 · 20261007</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#17223a;font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1700px;margin:auto;padding:22px}h1{font-size:25px;margin:0 0 8px}h2{font-size:20px;margin:16px 0 8px}h3{font-size:16px;margin:0 0 6px}p{margin:8px 0}.card{background:white;border:1px solid #dce2ec;border-radius:12px;padding:17px;margin:15px 0}.muted{color:#596780}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0;min-width:0}canvas,img{display:block;width:100%;height:auto;border-radius:6px;background:#e7ebf2}figcaption{font-weight:650;margin-bottom:5px}button,select{font:inherit;border:1px solid #ccd5e5;border-radius:7px;background:white;padding:6px 12px;cursor:pointer}button.active,.chosen{border:2px solid #2464d4;background:#eff5ff}button:hover{background:#edf3fd}.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.tag{background:#edf1f8;border-radius:5px;padding:2px 7px;font-size:13px}.warn{color:#ab4d00}.ok{color:#10775d}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid #e3e8f0;padding:8px;text-align:left;white-space:nowrap}tbody tr:hover{background:#f0f5ff}tbody tr.selected{background:#e9f1ff}tbody tr[data-case]{cursor:pointer}.overflow{overflow:auto}.chip{display:flex;gap:5px;align-items:center;border:1px solid #d7dfed;border-radius:6px;padding:3px 6px}.chip button{border:0;padding:2px 5px;font-size:13px}.dot{width:12px;height:12px;border-radius:3px;flex-shrink:0}details{margin-top:14px}summary{cursor:pointer;font-weight:650}.notice{background:#fff8e9;padding:12px;border-radius:7px}.tiny{font-size:12px}a{color:#245cbd}#seedImage{max-width:580px}.metrics{display:flex;gap:24px;flex-wrap:wrap}.metrics strong{font-size:20px}.legend{margin:10px 0}.frameNote{margin-top:9px}.error{color:#c53232}@media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style>
<main>
<h1>10 例修复验证</h1>
<p>你的选择已冻结：6 例局部 CropFormer、2 例 SAM2.1、1 例原始 CropFormer，共 55 个 mask；1 例不可靠，保留原结果。</p>
<div class="card"><p id="headline" class="notice"></p><div class="metrics" id="aggregate"></div><p class="muted tiny">每例独立从原始 3D 基线开始。变化是 10 例的均值，并非合并后场景总分。统一 v3 沿用当前调试预设；这是人工选择的修复可行性实验。</p>
<div class="overflow"><table><thead><tr><th>案例</th><th>选择</th><th>种子：目标 IoU 变化</th><th>追踪：目标 IoU 变化</th><th>追踪：场景 F1 变化</th><th>验收</th></tr></thead><tbody id="caseTable"></tbody></table></div>
<p class="row"><a href="http://127.0.0.1:8785/select.html" target="_blank">原三模型 mask 选择页</a><a href="../tracking_review_20261007/index.html" target="_blank">查看实际追踪动画与逐帧结果</a><a href="v3_summary.json" target="_blank">完整评估记录</a></p></div>
<section class="card" id="inspection"><h2 id="caseTitle"></h2><p id="choiceText"></p>
<div class="row" id="views"></div><p class="muted" id="frameText"></p>
<div class="row"><button data-mode="overlay" class="active">3D 实例叠加</button><button data-mode="changes">标签变化</button><button data-mode="rgb">RGB</button><button id="showAll">全部显示</button><button id="hideTargets">隐藏选中 mask</button><label><input id="selectedOnly" type="checkbox">仅看选中对象</label></div>
<div class="row legend" id="maskLegend"></div><p class="tiny muted">勾选可显示／隐藏单个 mask，点编号可单独看它。种子 mask 尚未形成 3D 归属时，对应的新 ID 会为空。</p>
<div class="grid" id="comparison"></div><p id="renderStatus" class="tiny muted"></p>
<p class="tiny muted">同一 RGB、视角、裁剪框和深度可见表面。灰色表示未发布；“标签变化”中绿色为新增、红色为撤回、橙色为换 ID。颜色保持一致，图中不使用 GT。</p>
<details id="seedDetails"><summary>所选种子与原三模型候选</summary><p id="seedCaption"></p><img id="seedImage" alt="按原始 mask 编号标出的所选种子"><div class="grid" id="candidates"></div></details>
<details open><summary>为什么新 mask 可能没有改变 3D 实例</summary><p>新旧证据使用相同帧票重；每帧先撤旧票再加入新票。短程追踪最多替换 9 个建图帧，其余历史帧的旧关联仍存在。</p><div class="overflow"><table><thead><tr><th>原 mask → 3D ID</th><th>关联</th><th>原主导票中位数</th><th>追踪后目标票中位数</th><th>种子表面符合该 ID</th></tr></thead><tbody id="auditTable"></tbody></table></div></details>
<details><summary>目标完整度、纯度与非目标误伤</summary><div id="targetMetrics"></div></details>
</section>
<div class="card"><h2>优先复核这 5 个 mask</h2><p>这些 mask 的可投影表面覆盖了两个较大的 GT 实例。请结合 RGB 查看是否仍合并了多个对象；只需复核这里列出的 mask。</p><p class="muted tiny">这是预测冻结后的诊断提示：第二个 GT 占投影点至少 20%。不等同于像素真值，原标注与本轮评分保持冻结。</p><div class="grid" id="seedFlags"></div></div>
<div class="card"><h2>下一步</h2><p>历史证据清单已生成，84%–98% 与种子身份不一致的旧票位于短程窗口之外。种子复核后，先在可信种子范围内重关联历史证据，再用同一批案例验证。保持每帧票重、几何和 v3 参数固定。</p><p class="row"><a href="history_summary.json" target="_blank">历史证据审计</a><a href="history_protocol.json" target="_blank">下一轮范围规则</a></p><p class="muted tiny">可信范围规则：只使用种子覆盖的表面，至少 2 个建图帧支持同一身份，追踪一致率 ≥80%。目前生成的是可回放的事务候选清单。已通过：7 场景投票／后处理复现、实际修复的重复执行与精确撤销、原始 CropFormer 种子零改动对照。严格验收要求目标 IoU 或结构错误有改善，同时不降低场景 F1/PQ、不损伤非目标对象。</p></div>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),cases=D.review.cases,S=D.summary.cases;
const idMaps=new Map(D.id_maps.cases.map(c=>[c.case_uid,c])),title={baseline:'修复前 · 原始 3D',seed_only:'仅替换种子帧',seed_plus_short_track:'种子＋短程追踪'};
const names={cropformer_local:'局部 CropFormer',sam2_1_local:'SAM2.1',original:'原始 CropFormer'};
let index=0,viewIndex=0,mode='overlay',hidden=new Set(),solo=null,epoch=0;
const $=id=>document.getElementById(id),E=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function pp(v){return v==null?'—':(v>=0?'+':'')+(v*100).toFixed(3)+' pp'}
function pc(v){return v==null?'—':(v*100).toFixed(2)+'%'}
const a=D.summary.aggregate;
$('headline').textContent=`首轮两组均为 ${a.seed_plus_short_track.strict_success_cases}/10 例通过严格验收；仅种子没有 AP50/F1 提升，短程追踪平均结果略降。`;
$('aggregate').innerHTML=`<div>种子帧 AP50 / F1<br><strong>${pp(a.seed_only.case_mean_scene_AP50_delta)} / ${pp(a.seed_only.case_mean_scene_F1_delta)}</strong></div><div>短程追踪 AP50 / F1<br><strong class="warn">${pp(a.seed_plus_short_track.case_mean_scene_AP50_delta)} / ${pp(a.seed_plus_short_track.case_mean_scene_F1_delta)}</strong></div>`;
$('caseTable').innerHTML=S.map((s,i)=>{const x=s.conditions.seed_only,y=s.conditions.seed_plus_short_track;return `<tr data-case="${i}"><td>${E(s.scene+' '+s.ROI)}</td><td>${E(names[s.choice]||'不可靠／跳过')}</td><td>${pp(x.target_mean_IoU_delta)}</td><td>${pp(y.target_mean_IoU_delta)}</td><td class="${y.F1_delta<0?'warn':''}">${pp(y.F1_delta)}</td><td>${s.status==='SKIP_UNRELIABLE'?'跳过':y.strict_success?'达标':y.F1_delta<0?'F1 下降／未达标':'未达标'}</td></tr>`}).join('');
$('caseTable').addEventListener('click',e=>{const r=e.target.closest('[data-case]');if(r){index=+r.dataset.case;viewIndex=0;hidden.clear();solo=null;$('selectedOnly').checked=false;showCase();$('inspection').scrollIntoView({behavior:'smooth'})}});
$('seedFlags').innerHTML=D.seed_review_flags.map(f=>`<figure><button data-review-uid="${E(f.case_uid)}">${E(f.scene+' '+f.ROI)} · mask ${f.native_mask_id}</button><p class="tiny muted">两个主要 GT 覆盖比：${f.major_GT_fractions.map(pc).join(' / ')}</p><img src="${f.image}" alt="${E(f.scene+' '+f.ROI)} 的 mask ${f.native_mask_id}" loading="lazy"></figure>`).join('');
$('seedFlags').onclick=e=>{const b=e.target.closest('[data-review-uid]');if(b){index=cases.findIndex(c=>c.case_uid===b.dataset.reviewUid);viewIndex=0;hidden.clear();solo=null;$('selectedOnly').checked=false;showCase();$('seedDetails').open=true;$('inspection').scrollIntoView({behavior:'smooth'})}};
function hsv(h){let i=Math.floor(h*6),f=h*6-i,p=.98*(1-.70),q=.98*(1-f*.70),t=.98*(1-(1-f)*.70);return [[.98,t,p],[q,.98,p],[p,.98,t],[p,q,.98],[t,p,.98],[.98,p,q]][i%6].map(v=>Math.floor(v*255))}
function color(id){const c=cases[index],row=c.objects.find(o=>o.persistent_id===id);return row?c.object_colors[row.track_id]:hsv((id*.61803398875)%1)}
function legend(){const c=cases[index];$('maskLegend').innerHTML=c.objects.map(o=>`<span class="chip"><input aria-label="显示 mask ${o.native_mask_id}" data-check="${o.persistent_id}" type="checkbox" ${hidden.has(o.persistent_id)?'':'checked'}><span class="dot" style="background:rgb(${color(o.persistent_id)})"></span><button data-solo="${o.persistent_id}" class="${solo===o.persistent_id?'active':''}">mask ${o.native_mask_id} ↔ 3D ${o.persistent_id}</button></span>`).join('')}
function showCase(){const c=cases[index],s=S[index],v=c.views[viewIndex];document.querySelectorAll('[data-case]').forEach(r=>r.classList.toggle('selected',+r.dataset.case===index));$('caseTitle').textContent=`${index+1}/10 · ${c.scene} ${c.ROI}`;$('choiceText').textContent=s.status==='SKIP_UNRELIABLE'?'标为不可靠：两组都保留原结果。':`你选的是 ${names[s.choice]}，包含 ${c.objects.length} 个 mask。目标结构错误本轮没有明显减少。`;
$('views').innerHTML=c.views.map((v,i)=>`<button data-view="${i}" class="${i===viewIndex?'active':''}">视角 ${v.number}${i===0?(s.status==='SKIP_UNRELIABLE'?' · 参考':' · 种子'):' · 校验'}</button>`).join('');$('frameText').innerHTML=`帧 ${v.frame} · 固定裁剪 [${v.box.join(', ')}] · <a href="${E(v.alarm_asset)}" target="_blank">原问题定位图</a>`;
legend();$('seedImage').hidden=!c.views[0].assets.human_seed;if(c.views[0].assets.human_seed)$('seedImage').src=c.views[0].assets.human_seed;$('seedCaption').textContent='所选种子使用原始 2D mask 编号。以下保留你标注时的三模型候选。';
$('candidates').innerHTML=['original','local','sam'].map((m,i)=>`<figure class="${['original','cropformer_local','sam2_1_local'][i]===s.choice?'chosen':''}"><figcaption>${['原始 CropFormer','局部 CropFormer','SAM2.1'][i]}</figcaption><img src="${c.candidate_images[m]}" alt="${m} 原候选" loading="lazy"></figure>`).join('');
$('auditTable').innerHTML=c.vote_audit.map(o=>`<tr><td>${o.native_mask_id} → ${o.persistent_id}</td><td>${o.association==='distinct_new_identity'?'新 ID':o.association==='exact_original_observation_control'?'原始对照':'匹配旧 ID'}</td><td>${o.baseline_leader_votes_median??'—'}</td><td>${o.conditions.seed_plus_short_track.desired_identity_votes_median??'—'}</td><td>${pc(o.conditions.seed_plus_short_track.desired_identity_is_final_fraction)}</td></tr>`).join('');
$('targetMetrics').innerHTML=['seed_only','seed_plus_short_track'].map(k=>{const x=s.conditions[k];return `<h3>${title[k]}</h3><p>原本正确的非目标实例变错：${x.previously_correct_unselected_lost}；非目标 IoU 下降超过 1 pp：${(x.unselected_IoU_degraded_over_0_01||[]).length}。</p><div class="overflow"><table><tr><th>GT ID</th><th>IoU 前 → 后</th><th>完整度前 → 后</th><th>纯度前 → 后</th></tr>${(x.target_objects||[]).map(r=>`<tr><td>${r.before.GT_id}</td><td>${pc(r.before.best_IoU)} → ${pc(r.after.best_IoU)}</td><td>${pc(r.before.completeness_best_single_instance)} → ${pc(r.after.completeness_best_single_instance)}</td><td>${pc(r.before.purity_of_best_IoU_instance)} → ${pc(r.after.purity_of_best_IoU_instance)}</td></tr>`).join('')}</table></div>`}).join('');
draw();}
const cache=new Map();function loadImg(path){if(!cache.has(path))cache.set(path,new Promise((res,rej)=>{const img=new Image();img.onload=()=>res(img);img.onerror=()=>rej(Error('图片加载失败：'+path));img.src=path}));return cache.get(path)}
async function draw(){const token=++epoch,c=cases[index],v=c.views[viewIndex],ids=idMaps.get(c.case_uid).views.find(x=>x.number===v.number).ID_assets,selected=new Set(c.objects.map(o=>o.persistent_id));$('comparison').innerHTML=Object.keys(title).map(k=>`<figure><figcaption>${title[k]}</figcaption><canvas id="canvas_${k}" aria-label="${title[k]}，帧 ${v.frame}"></canvas></figure>`).join('');$('renderStatus').textContent='正在载入当前视角…';try{
await Promise.all(Object.keys(title).map(async k=>{const path=mode==='changes'&&k!=='baseline'?v.assets[k+'_changes']:v.assets.RGB;const rgb=await loadImg(path);if(token!==epoch)return;const canvas=$('canvas_'+k);canvas.width=rgb.naturalWidth;canvas.height=rgb.naturalHeight;const ctx=canvas.getContext('2d');ctx.drawImage(rgb,0,0);if(mode!=='overlay')return;const idImg=await loadImg(ids[k]);if(token!==epoch)return;const tmp=document.createElement('canvas');tmp.width=canvas.width;tmp.height=canvas.height;const tc=tmp.getContext('2d');tc.drawImage(idImg,0,0);const decoded=tc.getImageData(0,0,tmp.width,tmp.height).data,im=ctx.getImageData(0,0,canvas.width,canvas.height),pixels=im.data,colors=new Map();
for(let q=0;q<decoded.length;q+=4){const code=decoded[q]|(decoded[q+1]<<8)|(decoded[q+2]<<16);if(code===0)continue;if(code===1){if(solo!==null||$('selectedOnly').checked)continue;for(let ch=0;ch<3;ch++)pixels[q+ch]=Math.round(pixels[q+ch]*.72+170*.28);continue}const id=code-1;if(hidden.has(id)||(solo!==null&&id!==solo)||($('selectedOnly').checked&&!selected.has(id)))continue;if(!colors.has(id))colors.set(id,color(id));const col=colors.get(id);for(let ch=0;ch<3;ch++)pixels[q+ch]=Math.round(pixels[q+ch]*.43+col[ch]*.57)}ctx.putImageData(im,0,0)}));if(token===epoch)$('renderStatus').textContent=mode==='overlay'?'3D mask 显隐仅影响显示；原预测与评估结果保持固定。':mode==='changes'?'仅标出真实 3D 标签变化；修复前面板显示 RGB。':'当前只显示 RGB。';
}catch(e){if(token===epoch){$('renderStatus').textContent=e.message+'；请通过本地 HTTP 页面打开。';$('renderStatus').className='error'}}}
$('views').onclick=e=>{const b=e.target.closest('[data-view]');if(b){viewIndex=+b.dataset.view;showCase()}};
$('maskLegend').onchange=e=>{if(e.target.dataset.check){const id=+e.target.dataset.check;e.target.checked?hidden.delete(id):hidden.add(id);legend();draw()}};
$('maskLegend').onclick=e=>{const b=e.target.closest('[data-solo]');if(b){const id=+b.dataset.solo;solo=solo===id?null:id;hidden.delete(id);legend();draw()}};
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));draw()});
$('showAll').onclick=()=>{hidden.clear();solo=null;$('selectedOnly').checked=false;legend();draw()};$('hideTargets').onclick=()=>{cases[index].objects.forEach(o=>hidden.add(o.persistent_id));solo=null;legend();draw()};$('selectedOnly').onchange=draw;
showCase();
</script></html>'''
(OUT / 'index.html').write_text(html.replace('__DATA__', data), encoding='utf-8')
(OUT / 'v3_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
for name in ['history_summary.json', 'history_protocol.json']:
    (OUT / name).write_bytes((HERE / 'server_reports' / name).read_bytes())
(OUT / 'seed_review_flags.json').write_text(json.dumps({'status': 'POST_HOC_REVIEW_ONLY', 'original_choices_modified': False, 'flags': flags}, ensure_ascii=False, indent=2), encoding='utf-8')
audit = [o for c in review['cases'] for o in c['vote_audit'] if o['association'] == 'distinct_new_identity']
digest = {'status': 'PASS', 'source_asset_roundtrip_hashes': len(hashes), 'review_cases': len(review['cases']),
          'views': sum(len(c['views']) for c in review['cases']), 'new_identity_masks': len(audit),
          'new_identity_masks_with_zero_final_seed_support': sum(o['conditions']['seed_plus_short_track']['desired_identity_is_final_fraction'] == 0 for o in audit),
          'source_export_sha256': '20ad6139511b71d146687d22d160fc7d8fd4baedc6d9cf6d0765ddf699681d61',
          'report': str(OUT / 'index.html')}
(HERE / 'delivery_summary.json').write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(digest, ensure_ascii=False))
