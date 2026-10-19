"""Immutable coverage statistics. Does not decide candidate scope or safety gates."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from PIL import Image
from coverage import measure

ORIGINAL=Path('/data/chenkejun/CVPR/results/instance_seed_grouping_20261015_r4')
INPUT=Path('/data/chenkejun/CVPR/results/seed_covisible_dedup_iou_only_20261018')
OUT=Path('/data/chenkejun/CVPR/results/seed_skip_repair_B_20261019')
BASE=Path('/data/chenkejun/CVPR/results/p1a1_one_vote_20261009/ablation/room2/abstain')
VIS=Path('/data/chenkejun/CVPR/results/overmerge_id_ratio_top20_20261015')
CODE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True,parents=True)
    assert not (OUT/'measurements.json').exists(),'Refuse to overwrite frozen measurements.'
    locked={}
    def lock(p,h=None):
        s=sha(p)
        if h is not None:assert s==h,('Input hash changed',str(p))
        locked[str(p)]=s
    for name,h in [('selection_binary.json','9989f8c47264d361b32593a1cb3c6c9702ad9609b99249a706bafc1461c770dc'),
                   ('seed_manifest.json','bf6576b177990584617b7e52e8e59f0a8b9192ac1f9480bfa8c4f2c32d23a17e'),
                   ('pair_results.json','f969cbbc96d268095781d6bdf0dae38dca4f96b99a28ee3ea803c8861fb59760')]:lock(INPUT/name,h)
    lock(ORIGINAL/'grouping.json','47d8ce6859e424c1cc5aa71f3f0a8f20eba8a62956977cf4f6e6630647812995')
    old=json.loads((ORIGINAL/'grouping.json').read_text());audit=json.loads((ORIGINAL/'audit.json').read_text())
    lock(ORIGINAL/'surface_supports.npz',audit['output_sha256']['surface_supports.npz'])
    lock(BASE/'final/instance_surface.npz','3f611dfe701a68259c71b94962ae6bae3c13fbac9c24f1e3eca5b414b8d2d4f3')
    lock(BASE/'surface_evidence.npz','b106b99741a4cb115d9ceed0ea3604ed46e0cc9a48d1feb865c2a0a60661c992')
    with np.load(BASE/'final/instance_surface.npz') as z:xyz=z['xyz_m'];labels=z['instance_id'];rgb=z['rgb']
    with np.load(BASE/'surface_evidence.npz') as z:
        assert np.array_equal(xyz,z['xyz_m']) and np.array_equal(rgb,z['rgb'])
        state=z['state']
    with np.load(ORIGINAL/'surface_supports.npz') as z:indices=z['indices'];offsets=z['offsets']
    visibility={};records=[]
    for s in old['seeds']:
        frame=s['frame']
        if frame not in visibility:
            p=VIS/f'frame{frame:06d}_visible.npz';lock(p,old['source_sha256'][str(p)])
            with np.load(p) as z:visibility[frame]=(z['vertex_index'],z['uv'])
        visible,uv=visibility[frame]
        a=s['index'];support=indices[offsets[a]:offsets[a+1]]
        assert len(support)==s['support_count'] and hashlib.sha256(support.tobytes()).hexdigest()==s['support_sha256']
        maskp=ORIGINAL/s['mask'];lock(maskp,s['mask_sha256']);mask=np.asarray(Image.open(maskp))>0
        assert np.array_equal(visible[mask[uv[:,1],uv[:,0]]],support)
        m=measure(support,visible,labels,state)
        records.append(dict(mask_id=s['id'],frame=frame,original_candidate_index=a,mask_sha256=s['mask_sha256'],**m))
    selection=json.loads((INPUT/'selection_binary.json').read_text());manifest=json.loads((INPUT/'seed_manifest.json').read_text())
    assert selection['counts']=={'retained':56,'excluded':16}
    bymask={r['mask_id']:r for r in records}
    primary=[]
    for g in manifest['groups']:
        m=bymask[g['primary']['id']]
        primary.append(dict(group_id=g['id'],original_track_id=g['original_track_id'],**m))
    assert len(primary)==56
    # This file is descriptive: only the already agreed p/r >= .95 threshold is evaluated.
    save('measurements.json',dict(schema='seed-repair-B-map-coverage-measurements-v1',
         status='measurements_only_no_skip_decision',input_seeds=56,candidate_measurements=len(records),
         source_frames=len(visibility),source_map=str(BASE/'final/instance_surface.npz'),
         current_instance_definition='final positive instance_id; unassigned points retained in S denominator',
         records=records,primaries=primary,elapsed_seconds=time.perf_counter()-started))
    for p,h in locked.items():assert sha(p)==h
    save('measurement_verification.json',dict(status='PASS',source_sha256=locked,
         measurements_sha256=sha(OUT/'measurements.json'),code_sha256={n:sha(CODE/n) for n in ['coverage.py','measure_inputs.py']},
         checks=['Original 56 identities and fixed pre-repair map hashes verified',
                 'Geometry aligned between final labels and native evidence',
                 '450 original mask/support identities verified and reconstructed from source visibility caches',
                 'S denominator includes unassigned points; I uses entire source-visible instance, not seed crop',
                 'No second-frame scope or conflict/sample gates chosen; no skip decisions exported']))
    passed=[r for r in primary if r['both95']]
    print(json.dumps(dict(candidates=len(records),primary_seeds=56,source_view_both95=len(passed),
         matches=[{k:r[k] for k in ['group_id','instance_id','support_count','p','r','native_state_counts']} for r in passed])))


if __name__=='__main__':main()
