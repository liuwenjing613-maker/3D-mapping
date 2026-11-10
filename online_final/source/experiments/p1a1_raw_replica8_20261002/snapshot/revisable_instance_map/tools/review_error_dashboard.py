#!/usr/bin/env python3
"""Audit cached diagnostic labels and rebuild readable case evidence; no map changes."""
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

ZH = dict(zip(
    'wall ceiling floor chair blinds sofa table rug window lamp door pillow bench tv-screen cabinet pillar blanket tv-stand cushion bin vent bed stool picture indoor-plant desk comforter nightstand shelf vase plant-stand basket plate monitor pipe panel desk-organizer wall-plug book box clock sculpture tissue-paper camera tablet pot bottle candle bowl cloth switch'.split(),
    '墙 天花板 地板 椅子 百叶窗 沙发 桌子 地毯 窗户 灯 门 枕头 长凳 电视屏幕 柜子 柱子 毯子 电视柜 靠垫 垃圾桶 通风口 床 凳子 装饰画 室内植物 书桌 被子 床头柜 架子 花瓶 花架 篮子 盘子 显示器 管道 面板 桌面收纳器 墙插 书 盒子 钟 雕塑 纸巾 相机 平板 盆 瓶子 蜡烛 碗 布 开关'.split()))


def audit_case_label(case, gt, classes):
    values = np.unique(gt.semantic_id[gt.instance_id == case['gt_id']])
    if len(values) != 1 or not 1 <= int(values[0]) <= len(classes):
        raise ValueError(f"Ambiguous GT semantics: {case['scene']} {case['gt_id']} {values}")
    semantic_id = int(values[0])
    name = classes[semantic_id - 1]
    case.update(class_name=name, class_name_zh=ZH.get(name, name),
                semantic_id=semantic_id, class_label_source='GT semantic_id, one-based; not a prediction class')


def dashboard_html(payload):
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    template = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P0 案例复核 · 标签已校正</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#142438;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}header,main{max-width:1440px;margin:auto;padding:20px 28px}h1{font-size:27px;margin:4px 0}h2{font-size:21px;margin:0 0 8px}h3{font-size:17px;margin:12px 0 6px}p{margin:8px 0}a{color:#1765b2}.muted,small{color:#52667d}.banner{background:#fff2cf;border:1px solid #dfc87f;border-radius:10px;padding:12px 16px}.guide,.panel{background:white;border:1px solid #dbe3ed;border-radius:12px;padding:20px;margin-bottom:16px}.guide ol{margin:6px 0;padding-left:22px}.layout{display:grid;grid-template-columns:295px minmax(0,1fr);gap:18px}aside{position:sticky;top:10px;align-self:start;max-height:92vh;overflow:auto}select,input,button{font:inherit;border:1px solid #bdcad9;border-radius:6px;padding:8px;background:white;color:#142438}select,input{width:100%;margin-bottom:8px}button{cursor:pointer}.case-button{width:100%;text-align:left;margin:5px 0;font-size:14px;line-height:1.5}.active{border:2px solid #187c81;background:#edfafa}.badge{display:inline-block;border-radius:6px;padding:2px 8px;background:#eaf0f7;font-size:13px;margin-right:7px}.fact{border-left:4px solid #187c81;padding:10px 15px;background:#f0f9f8}.hypothesis{border-left:4px solid #c08b23;padding:10px 15px;background:#fff9e9}.numbers{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.numbers>div{background:#edf3f8;padding:10px;border-radius:8px;font-size:13px}.numbers b{display:block;font-size:23px}figure{margin:12px 0}img{max-width:100%;width:100%;height:auto;object-fit:contain;border-radius:8px;background:#e9eef3}figcaption{font-size:14px;color:#435870;margin-bottom:6px}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:14px}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:5px}.frame{border-top:1px solid #dbe3ed;margin-top:20px;padding-top:10px}details{margin-top:12px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;max-height:300px;overflow:auto}dialog{width:95vw;max-width:1500px;max-height:95vh;border:0;border-radius:10px}dialog::backdrop{background:#000a}dialog img{width:100%}.zoom{padding:0;border:0;background:transparent;width:100%}.pagehint{font-size:14px}.evidence{padding-left:20px}.empty{padding:30px}.toolbar{display:flex;gap:12px;justify-content:space-between}@media(max-width:850px){header,main{padding:12px}.layout{grid-template-columns:1fr}aside{position:static;max-height:280px}.views{grid-template-columns:1fr}.numbers b{font-size:20px}}</style></head><body>
<header><h1>P0 无修复地图 · 案例复核</h1><p class="muted">已校正 GT 类别名称；逐帧观测与最终地图分开展示。</p><div class="banner"><b>这些案例来自历史 v1 评估（5 cm 映射），不是刚完成的 v3 错误清单。</b> FN/FP 和 IoU 均沿用 v1。<a href="ovimap_comparison.html">查看 v3 的 P0 / OVI-MAP 对照</a>。阶段原因是自动抽样线索，尚未通过干预实验确认。</div></header>
<main><section class="guide"><h2>先看什么</h2><ol><li><b>找对象：</b>看原始 RGB 的黄色定位框，确认 GT 标注的是哪个物体或物体部件。</li><li><b>看最终错误：</b>3D 图的黄色是目标漏掉的表面，粉色是该预测多带进来的表面，青色是二者重合。</li><li><b>往前追溯：</b>比较同一帧的黄色 GT 投影与绿色观测 mask。绿色只显示归属于标题中 Pred ID 的观测，不代表最终 3D 地图。</li></ol><p class="pagehint"><b>标签含义：</b>GT 是数据集对象编号；Pred 是系统持久实例编号，两者不需要相等。类别名称仅来自 GT，本系统此处按类别无关方式评估。显示的对象对是最大 IoU 对照，不表示匹配成功。FN 表示 GT 未匹配，FP 表示预测未匹配，同一对象对可能同时出现两次。</p></section>
<div class="layout"><aside class="panel"><label>场景<select id="scene"></select></label><label>错误类型<select id="type"><option value="">全部</option><option value="FN">GT 未匹配 · FN</option><option value="FP">预测未匹配 · FP</option></select></label><input id="search" placeholder="搜索类别 / GT / Pred"><small id="count"></small><div id="list"></div></aside><div id="detail"></div></div></main>
<dialog id="zoom"><button id="close">关闭大图</button><img id="large" alt="放大后的证据图片"></dialog>
<script>const DATA=__DATA__;const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const pct=n=>(100*Number(n||0)).toFixed(1)+'%';let current=0;
const names={surface_attribution_merge:'尚未定位：需检查表面归属和未抽样观测',borderline_overlap:'最终覆盖未达匹配条件（不一定是临界值）',spurious_or_small_fragment:'未匹配小区域（成因待核实）'};
const image=(path,alt)=>`<button class="zoom" data-src="${esc(path)}"><img loading="lazy" src="${esc(path)}" alt="${esc(alt)}"></button>`;
function facts(x){if(!x.iou)return '该对象对没有重叠。旧选择器在零 IoU 时选出的 Pred 只是占位，不能据此解释关联错误。';return `Pred ${esc(x.pred_uid)} 覆盖目标 GT 的 ${pct(x.recall)}；该预测中有 ${pct(1-x.precision)} 的表面不属于这个 GT。两者 IoU 为 ${x.iou.toFixed(3)}。`}
function evidence(x){const t=x.trace||{},p=t.prediction_lineage||{};let a=[];if(p.sampled_observations)a.push(`该 Pred 共 ${p.total_observations} 条观测，本次只抽查 ${p.sampled_observations} 条；其中 ${p.raw_mixed_observations} 条原始 mask 满足诊断混合阈值。`);if(t.raw_coverage!==undefined)a.push(`目标可见采样落入任意原始前景 mask 的比例为 ${pct(t.raw_coverage)}，细化后为 ${pct(t.final_coverage)}。这衡量前景支持，不等于分割正确率。`);if(t.persistent_ids?.length)a.push('目标投影收到的主要持久 ID 支持：'+t.persistent_ids.slice(0,4).map(z=>'Pred '+z[0]+'（'+z[1]+' 次采样）').join('、')+'。同一表面可能跨帧重复计数。');if(t.surface)a.push(`目标 GT 中 ${pct(t.surface.geometry_coverage)} 能找到 3 cm 内的 TSDF 点。低值也可能来自可见性、配准或几何偏移，需要进一步区分。`);return a.map(s=>'<li>'+esc(s)+'</li>').join('')||'<li>此案例尚无充分阶段证据。</li>'}
function show(i){current=i;const x=DATA.cases[i];const title=(x.class_name_zh||x.class_name)+' / '+x.class_name;const fn=x.error_type==='FN';const frames=x.review_frames||[];$('#detail').innerHTML=`<section class="panel"><div class="toolbar"><span class="badge">历史 v1 案例</span><span>${esc(x.scene)} · ${i+1}/${DATA.cases.length}</span></div><h2>${esc(title)} · GT ${x.gt_id}</h2><p><span class="badge">${fn?'GT 未匹配 · FN':'预测未匹配 · FP'}</span>对照 Pred ${esc(x.pred_uid||'无')}</p><p class="muted">${fn?'关注这个 GT 为什么没有得到完整、独立的实例。':'关注这个 Pred 为什么未能与任何 GT 成功匹配；标题类别仅表示对照 GT 的类别。'}</p><div class="fact"><b>已确认的最终现象</b><p>${facts(x)}</p></div><div class="numbers"><div><b>${x.iou.toFixed(3)}</b>IoU · 重叠 / 并集</div><div><b>${pct(x.precision)}</b>预测纯度 · 重叠 / Pred</div><div><b>${pct(x.recall)}</b>目标覆盖 · 重叠 / GT</div></div><h3>① 最终 3D：哪里漏了，哪里多了</h3><div class="legend"><span><i class="dot" style="background:#f6c85f"></i>黄色：GT 独有，目标未被此 Pred 覆盖</span><span><i class="dot" style="background:#ef5da8"></i>粉色：Pred 独有，不属于此 GT</span><span><i class="dot" style="background:#27d3c2"></i>青色：重合</span></div><figure>${image(x.plot_3d,'GT 与 Pred 的三维差异，两个观察角度')}<figcaption>同一对象对的两个固定角度；颜色表示差异，不是实例身份。点击图片可放大。点数：GT ${x.gt_size} / Pred ${x.pred_size}。</figcaption></figure><h3>② 原始画面与观测：先定位，再检查 mask</h3>${frames.map(f=>`<div class="frame"><b>帧 ${f.frame_id}</b><p class="muted">${esc(f.selection_reason)}。目标通过深度检查的可见投影：${f.visible_points} 点；归入 Pred ${esc(x.pred_uid)} 的本帧局部 mask ID：${esc(f.mask_local_ids.join(', ')||'无')}。</p><figure><figcaption>A · 原始 RGB：黄色框定位目标投影范围，框内不一定全是目标。</figcaption>${image(f.rgb,'原始 RGB，黄色框定位 GT 对象')}</figure><div class="views"><figure><figcaption>B · GT 参考：黄色点是该 GT 的可见投影；稀疏点不等于完整二维边界。</figcaption>${image(f.gt,'GT 深度可见投影')}</figure><figure><figcaption>C · 观测：绿色只属于 Pred ${esc(x.pred_uid)}。${f.mask_local_ids.length?'它是二维观测支持，不是最终三维分割。':'本帧没有归属于此 Pred 的观测，因此没有绿色。'}</figcaption>${image(f.pred,'关联到选中 Pred 的二维 mask')}</figure></div></div>`).join('')||'<p>没有足够的可见帧证据，不能把缺图解释成前端漏检。</p>'}<h3>③ 待验证的阶段线索</h3><div class="hypothesis"><b>${esc(names[x.root_cause]||DATA.cause_labels[x.root_cause]||x.root_cause)} · 自动线索</b><p>这不是已确认根因；请结合下面的抽样范围判断。${x.root_cause==='borderline_overlap'?'IoU 低于 0.5 不等于只差一点，旧分类名“临界失败”过于笼统。':''}</p></div><ul class="evidence">${evidence(x)}</ul><p><b>此案例建议先看：</b>${x.precision<.5?'检查绿色 mask 是否跨越多个真实对象；若单帧干净，再检查这些观测为何合到同一 Pred。':x.recall<.5?'检查黄色目标中未覆盖的部分是否有有效观测，再看它们被分配到其他 Pred、被细化删除，还是停留在不确定表面。':'检查黄色和粉色是否主要集中在边界，并核对是否受到几何映射距离影响。'}</p><details><summary>标签来源与原始追溯记录</summary><p>GT semantic_id=${x.semantic_id??'待核验'}；类别表采用从 1 开始的编号。本页不显示模型预测类别。</p><pre>${esc(JSON.stringify(x.trace,null,2))}</pre></details></section>`;document.querySelectorAll('.zoom[data-src]').forEach(b=>b.onclick=()=>{$('#large').src=b.dataset.src;$('#zoom').showModal()});renderList()}
function renderList(){const s=$('#scene').value,t=$('#type').value,q=$('#search').value.toLowerCase();let list=DATA.cases.map((x,i)=>({x,i})).filter(({x})=>(!s||x.scene===s)&&(!t||x.error_type===t)&&(!q||`${x.class_name} ${x.class_name_zh} ${x.gt_id} ${x.pred_uid}`.toLowerCase().includes(q)));$('#count').textContent=`${list.length} 个代表性案例；不是全部错误`;$('#list').innerHTML=list.map(({x,i})=>`<button class="case-button ${i===current?'active':''}" data-index="${i}"><b>${esc(x.class_name_zh||x.class_name)}</b> · ${x.error_type}<br>${esc(x.scene)} · GT ${x.gt_id} / Pred ${esc(x.pred_uid)}<br><small>IoU ${x.iou.toFixed(3)} · 目标覆盖 ${pct(x.recall)}</small></button>`).join('');$('#list').querySelectorAll('button').forEach(b=>b.onclick=()=>show(Number(b.dataset.index)));return list}
$('#scene').innerHTML='<option value="">全部场景</option>'+DATA.scenes.map(s=>`<option>${esc(s.scene_id)}</option>`).join('');$('#scene').value='room0';for(const id of ['scene','type','search'])$('#'+id).addEventListener(id==='search'?'input':'change',()=>{const rows=renderList();if(rows.length)show(rows[0].i);else $('#detail').innerHTML='<div class="panel">没有符合条件的案例。</div>'});$('#close').onclick=()=>$('#zoom').close();show(0);</script></body></html>'''
    from case_impact import add_impact_ui
    return add_impact_ui(template.replace('__DATA__', data))


def rebuild(run_root=None, output=None):
    import analyze_replica8_errors as base
    root = Path(run_root) if run_root else base.DATA_ROOT
    out = Path(output) if output else root / 'error_dashboard'
    payload = base.read_json(out / 'error_cases.json')
    # Retain the prior page and evidence manifest for audit.
    for name in ('index.html', 'error_cases.json'):
        backup = out / (name + '.before_label_fix')
        if not backup.exists():
            shutil.copy2(out / name, backup)
    original = base.read_json(out / 'error_cases.json.before_label_fix')
    old_names = {(x['scene'], x['error_type'], x['gt_id']): x['class_name'] for x in original['cases']}
    changed = []
    for scene in base.SCENES:
        sd = root / scene
        gt = base.load_gt(sd / 'ground_truth/gt.npz')
        classes = base.read_json(base.REFERENCE_ROOT / scene / 'manifest.json')['semantic_classes']
        config = base.read_json(sd / 'configs/p0_parent_regrouped.json')
        src = Path(config['source']['scene_root'])
        masks = Path(config['source']['mask_root'])
        poses = np.loadtxt(src / config['source']['trajectory']).reshape(-1, 4, 4)
        cam = config['camera']
        camera = (cam['fx'], cam['fy'], cam['cx'], cam['cy'])
        lineage, _ = base.load_lineage(sd)
        for x in [c for c in payload['cases'] if c['scene'] == scene]:
            old = old_names[(scene, x['error_type'], x['gt_id'])]
            audit_case_label(x, gt, classes)
            changed.append({'scene':scene, 'gt_id':x['gt_id'], 'before':old, 'after':x['class_name']})
            if x['error_type'] == 'FN' and x['iou'] == 0:
                x.setdefault('zero_overlap_placeholder_removed', x['pred_uid'])
                x['pred_uid'] = ''
                x['pred_size'] = 0
                x.get('trace', {}).pop('prediction_lineage', None)
                indices = np.flatnonzero(gt.instance_id == x['gt_id'])
                base.save_3d_plot(out / x['plot_3d'], gt.xyz_ref, indices,
                                  np.empty(0, np.int32), f"{scene} GT {x['gt_id']} | no overlapping prediction")
            points = gt.xyz_ref[gt.instance_id == x['gt_id']]
            points = points[base.downsample(np.arange(len(points)), 12000)]
            instance = int(x['pred_uid']) if x['pred_uid'].isdigit() else -1
            obs = lineage.get(instance, [])
            if 'birth_decision' in x.get('trace', {}) and obs:
                first = min(obs, key=lambda r:int(r['frame_id']))
                x['trace']['birth_decision'] = first['decision']
                x['trace']['birth_frame'] = int(first['frame_id'])
            frames = []
            for image in x['frame_images']:
                f = int(re.search(r'_f(\d+)\.jpg$', image).group(1))
                rgb = cv2.imread(str(src / config['source']['rgb_pattern'].format(frame=f)))
                depth = cv2.imread(str(src / config['source']['depth_pattern'].format(frame=f)), -1).astype(np.float32)/cam['depth_png_units_per_meter']
                mask = cv2.imread(str(masks / config['source']['mask_pattern'].format(frame=f)), -1)
                _, u, v = base.project_points(points, poses[f], camera, depth)
                ids = sorted({int(o['mask_local_id']) for o in obs if int(o['frame_id']) == f})
                support = np.isin(mask, ids) if ids else np.zeros(mask.shape, bool)
                gt_view, pred_view, context = rgb.copy(), rgb.copy(), rgb.copy()
                for a,b in zip(u[::max(1,len(u)//2500)],v[::max(1,len(v)//2500)]):
                    cv2.circle(gt_view,(int(a),int(b)),2,(0,210,255),-1)
                pred_view[support]=(0.55*pred_view[support]+0.45*np.array([60,220,90])).astype(np.uint8)
                contours,_=cv2.findContours(support.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(pred_view,contours,-1,(60,255,120),2)
                ys,xs=np.nonzero(support)
                xs=np.r_[xs,u];ys=np.r_[ys,v]
                box=(0,0,rgb.shape[1],rgb.shape[0])
                if len(xs):
                    box=(max(0,int(xs.min())-50),max(0,int(ys.min())-50),min(rgb.shape[1],int(xs.max())+51),min(rgb.shape[0],int(ys.max())+51))
                if len(u):
                    cv2.rectangle(context,(int(u.min()),int(v.min())),(int(u.max()),int(v.max())),(0,210,255),3)
                x0,y0,x1,y1=box
                stem=Path(image).stem+'_review'
                paths={}
                for kind,array in [('rgb',context),('gt',gt_view[y0:y1,x0:x1]),('pred',pred_view[y0:y1,x0:x1])]:
                    rel=f'assets/{stem}_{kind}.jpg'
                    if not cv2.imwrite(str(out/rel),array,[cv2.IMWRITE_JPEG_QUALITY,93]):raise OSError(rel)
                    paths[kind]=rel
                frames.append(dict(frame_id=f,visible_points=len(u),mask_local_ids=ids,**paths,
                    selection_reason='按目标 GT 可见采样数量选取' if x['error_type']=='FN' else '按该 Pred 的观测有效像素数量选取'))
            x['review_frames']=frames
        print(scene,'labels and frame panels rebuilt',flush=True)
    payload['review_version']=2
    payload['protocol']='Replica-CA-v1 historical case selection'
    from case_impact import enrich
    enrich(payload, root)
    base.write_json(out/'error_cases.json',payload)
    (out/'index.html').write_text(dashboard_html(payload),encoding='utf-8')
    base.write_json(out/'label_audit.json',{'cases_checked':len(changed),'labels_changed':sum(r['before']!=r['after'] for r in changed),'mapping':'GT semantic_id -> semantic_classes[id - 1]','cases':changed})
    print('DONE',len(changed),flush=True)


if __name__=='__main__':
    rebuild()
