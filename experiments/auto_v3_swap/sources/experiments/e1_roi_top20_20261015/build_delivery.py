from pathlib import Path
import subprocess,sys,shutil,json,hashlib
H=Path(__file__).resolve().parent;F=H.parent/'overmerge_id_ratio_top20_20261015';ROOT=H.parents[1]/'results/固定案例_三模型对比_20261006'
oldfiles=[ROOT/d/n for d in ['room2_E1_retained_ROI_20261014','room2_E1_prerepair_CUT_top10_20261014'] for n in ['index.html','viewer_data.json']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
oldhash={str(p):sha(p) for p in oldfiles}
def run(path):subprocess.run([sys.executable,str(path)],check=True)
for name in ['verify_expansion.py','unpack_verify.py','build_page.py','verify_delivery.py']:run(H/name)
audit=ROOT/'room2_E1_overmerge_audit_top20_20261015';audit.mkdir(exist_ok=True);shutil.copyfile(F/'data.json',audit/'data.json')
for name in ['verify_filter_expansion.py','build_retained.py','add_rejected_toggle.py','verify_retained.py']:run(F/name)
subprocess.run(['node','--check',str(F/'retained_runtime.js')],check=True)
assert oldhash=={str(p):sha(p) for p in oldfiles}
stats=json.loads((F/'data.json').read_text(encoding='utf-8'))['stats']
note=f'''room2 / E1 修复前 / 三状态各前 20 / 2026-10-15
网页：http://127.0.0.1:8785/room2_E1_retained_ROI_top20_20261015/index.html
三类各按绝对点数选择前 20，共 60 个入榜名额，去重为 32 个 ROI，180 个历史视角条目，112 个物理帧，630 个裁剪。原有 412 个裁剪全部保留，新增 218 个。
三种分割设置不变：原图 CropFormer 冻结缓存、同裁剪局部 CropFormer、SAM2.1 自动分割。已有相同输入的结果按哈希复用。
筛选规则不变：N 是完整候选内有效可见的固定 TSDF 顶点数；n2 是第二大原生 CONFIRMED 持久身份的支持点数。仅当 n2>=10 且 5*n2>=N 时拟剔除。U/T/CONFLICT 计入分母，不作为独立物体。N=0 保留为无法判断。
去重候选 {stats['unique']}，拟剔除 {stats['reject']}，其余 {stats['keep']+stats['no_support']}，其中无有效支持 {stats['no_support']}。这不是人工确认的过合并数量。
旧 ROI 三栏检查格式；拟剔除默认隐藏，可整体或逐栏显示，红色边界与“拟剔除 ID”标识。拟剔除为对照显示，不加入合格目标。
新存储键：cvpr-room2-e1-retained-roi-top20-20261015-v1；?qa=1 使用独立测试键。原页面和记录未改动。请导出质量判断 JSON。
校验：expansion_verification.json、filter_expansion_verification.json、retained_verification.json。地图未修改；本轮没有执行修复或评价，GT 未用于生成或筛选。后续评价仍使用 main 当前 object_observed_repair / revision 2。
服务器代码：/home/chenkejun/CVPR/experiments/e1_roi_top20_20261015 与 overmerge_id_ratio_top20_20261015。
服务器数据：/data/chenkejun/CVPR/results/e1_roi_top20_20261015 与 overmerge_id_ratio_top20_20261015。
'''
(H/'README.txt').write_text(note,encoding='utf-8');(F/'README.txt').write_text(note,encoding='utf-8')
print('DELIVERY_READY http://127.0.0.1:8785/room2_E1_retained_ROI_top20_20261015/index.html',flush=True)
