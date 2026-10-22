"""Portable paired trial review, real projections and three exact PLY states."""
from pathlib import Path
from html import escape
import hashlib,json,shutil,tarfile
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).parent
ROOT=HERE.parents[1]
DELIVERY=ROOT/'results/固定案例_三模型对比_20261006'
OUT=DELIVERY/'joint_mask_replacement_20261007'
BUNDLE=HERE/'joint_review_bundle.tar.gz'

def main():
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='e5b2acd85eb4d24b3a5b08a07b23ffc5cd36a07f8a0badce4cac9690c0c13ca1'
    with tarfile.open(BUNDLE) as tar:
        for item in tar.getmembers():
            p=(DELIVERY/item.name).resolve()
            assert p.is_relative_to(OUT.resolve()) and not item.issym() and not item.islnk()
        tar.extractall(DELIVERY,filter='data')
    data=json.loads((OUT/'review_data.json').read_text(encoding='utf-8'))
    s=data['summary'];d=data['frame_decisions']
    source=HERE/'server_reports'
    for name in ['joint_v3_summary.json','joint_repair_summary.json','joint_frame_decisions.json','joint_evaluation_freeze.json','joint_tracking_status.json']:
        shutil.copy2(source/name,OUT/name)
    assert hashlib.sha256((HERE/'joint_mask_policy.json').read_bytes()).hexdigest()==s['policy_sha256']
    assert hashlib.sha256((ROOT/'修复候选选择_20261006.json').read_bytes()).hexdigest()=='20ad6139511b71d146687d22d160fc7d8fd4baedc6d9cf6d0765ddf699681d61'
    shutil.copytree(DELIVERY/'tracking_review_20261007/vendor',OUT/'vendor',dirs_exist_ok=True)
    shutil.copy2(HERE/'mask_visibility.js',OUT/'mask_visibility.js')
    shutil.copy2(DELIVERY/'tracking_review_20261007/instance_palettes.json',OUT/'instance_palettes.json')
    model=data['model']
    model['seed_union_bounds']=[
        [min(o['seed_surface_bounds'][0][k] for o in model['objects']) for k in range(3)],
        [max(o['seed_surface_bounds'][1][k] for o in model['objects']) for k in range(3)]]
    manifest=json.loads((OUT/'ply_models/manifest.json').read_text(encoding='utf-8'))
    manifest['cases'][0]['seed_union_bounds']=model['seed_union_bounds']
    (OUT/'ply_models/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    js=(HERE/'floating_ply.js').read_text(encoding='utf-8')
    js=js.replace("const states=['baseline','seed_only','seed_plus_short_track'];","const states=['baseline','pixel_control','whole_mask'];")
    js=js.replace("const stateNames=['P1-A1 原版（holes_geodesic）','仅种子修复','短程追踪修复'];", "const stateNames=['原基线','局部像素换票对照','整张旧 mask 换票'];")
    js=js.replace("return o?.seed_surface_bounds||model?.local_bounds;", "return o?.seed_surface_bounds||model?.seed_union_bounds||model?.local_bounds;")
    js=js.replace('model.label_changes.seed_plus_short_track','model.label_changes.whole_mask')
    js=js.replace('追踪修复改变','整张 mask 换票改变')
    (OUT/'floating_ply.js').write_text(js,encoding='utf-8')
    shutil.copy2(HERE/'floating_ply.css',OUT/'floating_ply.css')
    float_html=(HERE/'floating_ply.html').read_text(encoding='utf-8')
    float_html=float_html.replace('__PLY_DATA__',(OUT/'ply_models/manifest.json').read_text(encoding='utf-8').replace('</','<\\/'))
    float_html=float_html.replace('仅种子','局部换票对照').replace('追踪修复后','整张 mask 换票').replace('本轮短程修复的最终状态','本次整张旧 mask 换票试验的最终状态')
    float_html=float_html.replace('修复后 3D PLY','双椅子换票 · 3D PLY').replace('floating_ply.js?v=all-instances-20261007','floating_ply.js?v=joint-v1')
    float_html=float_html.replace('floating_ply.js?v=visibility-fixed2-20261007','floating_ply.js?v=joint-v2')
    float_html=float_html.replace('<option value="3">与 2D mask 同色</option>','<option value="3" selected>与 2D mask 同色</option>')
    float_html=float_html.replace('默认保留原 PLY 配色。点击 mask 只定位；取消勾选才隐藏对应 ID。','蓝色 ID52、绿色 ID355，其余实例保留原 PLY 配色。点击 mask 定位；取消勾选隐藏对应 ID。')
    floating_note='实例按 ID 彩色显示，灰色表示未发布。黄色框仅指示种子覆盖范围。变化着色：绿＝新增，红＝撤回，橙＝换 ID；点云为本次整张旧 mask 换票试验的最终状态。'
    float_html=float_html.replace(floating_note,'所有有效实例彩色显示。黄色框表示种子覆盖范围；变化着色：绿＝新增，红＝撤回，橙＝换 ID。三组使用同一份 TSDF 几何。')
    baseline=s['baseline_metrics']
    states={'baseline':baseline,**{a:v['metrics'] for a,v in s['arms'].items()}}
    names={'baseline':'原基线','pixel_control':'局部像素换票对照','whole_mask':'整张旧 mask 换票'}
    table=''.join('<tr><td>'+names[a]+'</td>'+''.join(f'<td>{m[k]*100:.2f}%</td>' for k in ['CA_AP_uniform','CA_AP50_uniform'])+f'<td>{m["CA_PRF1_0_5"]["F1"]*100:.2f}%</td><td>{m["CA_PQ"]["PQ"]*100:.2f}%</td></tr>' for a,m in states.items())
    rows=[]
    for mid,gtid in [(14,4001),(24,4002)]:
        b=next(x['before'] for x in s['arms']['whole_mask']['target_objects'] if x['before']['GT_id']==gtid)
        values=[b['best_IoU']]+[next(x['after']['best_IoU'] for x in s['arms'][a]['target_objects'] if x['before']['GT_id']==gtid) for a in ['pixel_control','whole_mask']]
        rows.append(f'<tr><td>mask {mid}</td>'+''.join(f'<td>{v*100:.2f}%</td>' for v in values)+f'<td>{s["surface_geometry_1cm_reachability_upper_bound_by_GT"][str(gtid)]*100:.2f}%</td></tr>')
    figures=[]
    for v in data['views']:
        figures.append(f'<section><h2>固定视角 f{v["frame"]} · '+('已用于换票' if v['decision']['accepted'] else '检查未通过，保留旧证据')+'</h2><div class="views">'+''.join(f'<figure><figcaption>{names[a]}</figcaption><a href="{v["assets"][a]}" target="_blank"><img src="{v["assets"][a]}" alt="f{v["frame"]} {names[a]} 实际3D标签投影"></a></figure>' for a in states)+'</div></section>')
    related=sum(bool(x['old_local_masks']) for x in d['frames'])
    retired=len(d['retired_observations'])
    intervals=[]
    for fid in d['accepted_frames']:
        if intervals and fid==intervals[-1][1]+5:intervals[-1][1]=fid
        else:intervals.append([fid,fid])
    c={'case_uid':'room2-14047feb0df3aaa5','scene':'room2','ROI':'mask14 + mask24',
       'objects':[{'track_id':4,'native_mask_id':14,'audit':{'persistent_id':52}}, {'track_id':7,'native_mask_id':24,'audit':{'persistent_id':355}}],
       'object_colors':{'4':[40,186,245],'7':[64,234,120]}}
    script='''const c=__CASE__;let solo=null;const visible=new Set([4,7]);
window.pilotPlyApi={state:()=>({c,solo,visible_track_ids:[...visible],enabled_track_ids:[...visible]})};
function sync(){window.pilotPlyViewer?.sync(window.pilotPlyApi.state());}
document.querySelectorAll('[data-target]').forEach(b=>b.onclick=()=>{solo=b.dataset.target==='all'?null:+b.dataset.target;document.querySelectorAll('[data-target]').forEach(x=>x.classList.toggle('active',x===b));sync();});
document.querySelectorAll('[data-visible]').forEach(box=>box.onchange=()=>{const id=+box.dataset.visible;box.checked?visible.add(id):visible.delete(id);sync();});
'''.replace('__CASE__',json.dumps(c,ensure_ascii=False))
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双椅子整张 mask 换票 · 实际结果</title><link rel="stylesheet" href="floating_ply.css"><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5fa;color:#172b46;font:16px/1.65 system-ui,"Microsoft Yahei",sans-serif}main{max-width:1240px;margin:auto;padding:26px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}section,.hero{background:white;padding:22px;border-radius:14px;margin:18px 0;box-shadow:0 2px 12px #18355309}.bad{color:#ae3434;font-weight:700}.ok{color:#17754c;font-weight:700}.pill{display:inline-block;background:#e8edf6;padding:5px 12px;border-radius:20px;margin:4px 8px 4px 0}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #dce4ef}th{background:#edf3fa}.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}figcaption{font-weight:600;margin-bottom:8px}img{width:100%;border-radius:8px}button{padding:9px 14px;border:1px solid #aebcd1;background:white;border-radius:7px;cursor:pointer;margin:5px}.active{background:#e3f1ff;border-color:#2284e4}a{color:#166bab}.muted{color:#60738c;font-size:14px}summary{cursor:pointer}#plyFloat{width:560px;height:560px}@media(max-width:760px){.views{grid-template-columns:1fr}main{padding:12px}}</style><main>
<div class="hero"><h1>双椅子整张 mask 换票 · 实际试验</h1><div class="ok">换票正确性校验：通过</div><div class="bad">修复效果验收：未通过</div><p>room2 的 mask14、24 单独联合追踪，原始种子不变。两组使用同一批可靠帧和同一份新 mask，只改变旧 mask 投票的撤回范围。</p>
<span class="pill">完整追踪 2000 帧 / 3分11秒</span><span class="pill">换票 __ACCEPT__/__RELATED__ 个有旧观察的建图帧</span><span class="pill">撤回 __RETIRED__ 个旧 mask 观察</span><p class="muted">统一 v3 当前调试配置，非正式评分。全部 400 个建图帧参与最终累加；未通过检查的帧继承原证据。仅代表这两个目标。</p></div>
<section><h2>统一 v3：整场景指标</h2><table><tr><th>状态</th><th>AP</th><th>AP50</th><th>F1</th><th>PQ</th></tr>__METRICS__</table></section>
<section><h2>两把椅子的完整实例 IoU</h2><table><tr><th>目标</th><th>基线</th><th>局部换票</th><th>整张 mask 换票</th><th>当前几何适配上限</th></tr>__TARGETS__</table><p>原来的两张 2D mask 各自独立，但都归入 3D ID52。换票后产生了 ID52、ID355；mask24 新 ID 的纯度为99.94%，完整目标覆盖率仅17.71%，仍有旧 ID 残留。种子表面正确归属率从对照组82.55%升到85.13%，不能据此判定完整实例修复成功。</p><p class="muted">几何上限由评估完成后的独立诊断得到：所有 TSDF 顶点理想分配给目标时，1cm 距离规则可匹配的 GT 点比例。该数值未用于筛帧或修复，也未改变 v3 阈值。它不等于真实网格缺失比例。<a href="geometry_diagnostic.json">查看诊断记录</a></p></section>
<section><h2>悬浮 3D PLY 操作</h2><button data-target="all" class="active">查看两个实例</button><button data-target="4">定位 mask14 ↔ ID52</button><button data-target="7">定位 mask24 ↔ ID355</button><label><input type="checkbox" data-visible="4" checked>显示 ID52</label> <label><input type="checkbox" data-visible="7" checked>显示 ID355</label><p class="muted">右下角是真实 PLY，可旋转、缩放、移动和切换基线／局部换票／整张 mask 换票。蓝色 ID52、绿色 ID355；黄色框是种子范围。下方图片使用相同几何、相同深度可见性和原先固定的三个视角。</p></section>
__FIGURES__
<section><h2>正确性和来源记录</h2><p class="ok">400 帧完整重算与增量换票账本逐项一致；撤回试验恢复原票数完全一致；同帧同点同 ID 去重；未替换旧 mask 的表面支持保持不变；原始标注、基线和 TSDF 几何均未修改。</p><p>已替换建图帧区间：__INTERVALS__。其余相关帧保留旧证据，仍需要确认。</p><details><summary>查看筛帧规则与失败原因</summary><p>每个目标至少20个深度可见种子点、覆盖率≥80%；新表面支持在目标种子包围盒内比例≥90%；旧家族支持在联合包围盒内≥90%；保护其他种子和旧实例。具体规则在运行与评分前固定。</p><a href="joint_frame_decisions.json">逐帧判断与旧 mask 撤回清单</a></details><p><a href="joint_v3_summary.json">完整 v3 结果</a> · <a href="joint_repair_summary.json">两组账本和回滚校验</a> · <a href="joint_evaluation_freeze.json">评分前冻结记录</a> · <a href="ply_models/joint_full.ply">下载完整 PLY</a></p></section></main>
__FLOAT__<script>__SCRIPT__</script></html>'''
    for token,value in [('__ACCEPT__',str(s['accepted_mapping_frames'])),('__RELATED__',str(related)),('__RETIRED__',str(retired)),('__METRICS__',table),('__TARGETS__',''.join(rows)),('__FIGURES__',''.join(figures)),('__INTERVALS__',escape(', '.join(f'{a}–{b}' for a,b in intervals))),('__FLOAT__',float_html),('__SCRIPT__',script)]:html=html.replace(token,value)
    (OUT/'index.html').write_text(html,encoding='utf-8')
    # Static scientific contact sheet with the same fixed views and label states.
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    sheet=Image.new('RGB',(1500,len(data['views'])*365+90),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,20),'mask14、24 联合换票：相同 TSDF 几何与固定视角',font=font,fill='#172b46')
    for row,v in enumerate(data['views']):
        for col,a in enumerate(states):
            image=Image.open(OUT/v['assets'][a]);image.thumbnail((490,315))
            x,y=col*500,row*365+90
            draw.text((x+12,y),f'f{v["frame"]} · {names[a]}',font=font,fill='#172b46')
            sheet.paste(image,(x+(500-image.width)//2,y+43))
    sheet.save(OUT/'joint_actual_comparison.jpg',quality=95)
    proof={'status':'PASS','source_bundle_sha256':hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
        'native_geometry_and_PLY_label_roundtrips_verified':True,'full_scene_points':data['model']['full']['points'],
        'local_points':data['model']['local']['points'],'old_observations_retired':retired,
        'accepted_frames':s['accepted_mapping_frames'],'related_frames':related,'strict_success':False,
        'all_original_annotation_bytes_unchanged':True,'same_fixed_views_and_visibility':True}
    (OUT/'verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False))

if __name__=='__main__':main()
