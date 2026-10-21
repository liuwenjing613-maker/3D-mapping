"""Score the frozen point-replacement stages through main's default R2 CLI."""
from bootstrap import C,R,SOURCE,PARTIAL,COUNT,SCENE,GROUP,MAIN,verify_sources
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
import numpy as np
from unified_eval.current_protocol import CURRENT_PROTOCOL,CURRENT_CONFIG,REGISTRY_PATH,require_current_protocol,require_current_gt
from unified_eval.io import load_gt,sha256_file
from unified_eval.geometry import load_fixed_surface_correspondence
read=lambda p:json.loads(Path(p).read_text());sha=sha256_file
OUT=R/'evaluation_r2';STAGES=['direct','core_all','e1']
LOCKS=['geometry_xyz_sha256','correspondence_sha256','gt_scope_sha256','gt_observed_support_sha256',
       'profile_config_sha256','profile_semantics_sha256','evaluator_code_sha256']

def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(p)+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def cli(args,log):
    with log.open('w') as stream:p=subprocess.run([sys.executable,'-B','-m','unified_eval.cli',*map(str,args)],cwd=MAIN,stdout=stream,stderr=subprocess.STDOUT)
    assert p.returncode==0,str(log)

def main():
    started=time.time();verify_sources();assert require_current_protocol(CURRENT_CONFIG)
    assert CURRENT_PROTOCOL['evaluable_GT_count']==350 and CURRENT_PROTOCOL['frozen'] is False
    freeze=read(R/'predictions_freeze.json');assert freeze['status']=='PREDICTIONS_FROZEN_BEFORE_GT' and not freeze['GT_read_by_mapper']
    for kind in ['code_sha256','source_sha256','output_sha256']:
        for p,d in freeze[kind].items():assert sha(p)==d,p
    OUT.mkdir(exist_ok=True)
    gtpath=Path(CURRENT_PROTOCOL['canonical_GT'][SCENE]['file']);assert require_current_gt(gtpath)==SCENE
    gt=load_gt(gtpath)
    materialpath=Path(f'/data/chenkejun/CVPR/revisable_instance_map/p1a1_raw_replica8_20261002/native/{SCENE}/P1-A1/surface_p0/materialization_report.json')
    associationpath=materialpath.parent.parent/'association/association_report.json'
    material,association=read(materialpath),read(associationpath)
    frames=[x['frame_id'] for x in material['frame_records']]
    assert frames==[x['frame_id'] for x in association['frame_summaries']]==gt.metadata['observation_metadata']['frame_ids']==list(range(0,2000,5))
    fsha=hashlib.sha256(json.dumps(frames,separators=(',',':')).encode()).hexdigest()
    assert fsha==gt.metadata['observation_metadata']['frame_list_sha256']
    config=Path(f'/data/chenkejun/CVPR/revisable_instance_map/replica8_p0_main/{SCENE}/configs/raw.json')
    assert sha(config)==material['config_sha256']==association['config_sha256']
    cache_root=Path('/data/chenkejun/CVPR/results/p1a1_strict_repair_20261009/evaluation_r2/geometry_cache') if SCENE in ['room0','room2'] else Path('/data/chenkejun/CVPR/results/auto_v2_replica6_20261019/geometry_cache')
    cache=cache_root/SCENE/'fixed_gt_to_tsdf_r2.npz'
    assert sha(cache)==read(SOURCE/'evaluation_r2/protocol_freeze.json')['fixed_correspondence_sha256']
    dump(OUT/'protocol_freeze.json',dict(current_protocol=CURRENT_PROTOCOL,registry_sha256=sha(REGISTRY_PATH),default_main_CLI=True,
        entry_commit=subprocess.check_output(['git','-C',str(MAIN),'rev-parse','HEAD'],text=True).strip(),fixed_correspondence_sha256=sha(cache)))
    oldeval=PARTIAL/'evaluation_r2';oldrows=read(oldeval/'comparison.json')['rows'];rows=[];evals={};manifests={}
    for stage in ['native','final']:
        ref=read(oldeval/f'baseline_{stage}/adapter_manifest.json')
        reference=np.load(ref['source_surface'])
        for name,old in [('baseline','baseline'),('whole','old_auto_v2'),('partial','whole_to_partial')]:
            path=oldeval/f'{old}_{stage}';m=read(path/'adapter_manifest.json')
            assert sha(m['source_surface'])==m['source_surface_sha256']
            for k in LOCKS:assert m[k]==ref[k],(name,stage,k)
            rows.append(dict(next(x for x in oldrows if x['variant']==old and x['stage']==stage),variant=name,selected=0 if name=='baseline' else COUNT))
            evals[name,stage]=path;manifests[name+'_'+stage]=m
        for name in STAGES:
            surface=R/name/SCENE/stage/'instance_surface.npz';dest=OUT/f'{name}_{stage}'
            z=np.load(surface)
            np.testing.assert_array_equal(z['xyz_m'],reference['xyz_m']);np.testing.assert_array_equal(z['rgb'],reference['rgb'])
            load_fixed_surface_correspondence(cache,z['xyz_m'],gt.xyz_ref)
            provenance=dict(input_scope_verified=True,frame_list_sha256=fsha,frame_count=400,input_config_sha256=sha(config),
                materialization_report_sha256=sha(materialpath),association_report_sha256=sha(associationpath),
                prediction_freeze_sha256=sha(R/'predictions_freeze.json'),prediction_frozen_before_GT=True,GT_used_for_mapping=False,
                condition=name,selection_group=GROUP,selected_primary_masks=COUNT,batch=str(R),surface_sha256=sha(surface))
            pp=OUT/'provenance'/f'{name}_{stage}.json';dump(pp,provenance)
            if not (dest/'metrics.json').exists():
                cli(['adapt-surface','--gt',gtpath,'--surface',surface,'--fixed-correspondence',cache,'--source-provenance',pp,
                     '--method-name',f'point_replace_{GROUP}_{SCENE}_{name}_{stage}','--method-commit',sha(C/'replay.py'),'--out',dest],OUT/f'{name}_{stage}.log')
            m=read(dest/'adapter_manifest.json');assert m['source_surface_sha256']==sha(surface)
            for k in LOCKS:assert m[k]==ref[k],(name,stage,k)
            sm=read(dest/'metrics.json')['summary'];e=np.load(R/name/SCENE/'native/surface_evidence.npz');lab=z['instance_id']
            row=dict(variant=name,stage=stage,selected=COUNT,F1=sm['CA_PRF1_0_5']['F1'],PQ=sm['CA_PQ']['PQ'],AP=sm['CA_AP_uniform'],AP50=sm['CA_AP50_uniform'],mCov=sm['CA_mCov'],
                TP=sm['CA_PRF1_0_5']['TP'],FP=sm['CA_PRF1_0_5']['FP'],FN=sm['CA_PRF1_0_5']['FN'],gray_total=int((lab<=0).sum()),
                gray_states={n:int(((lab<=0)&(e['state']==i)).sum()) for n,i in [('U',0),('T',1),('C',3)]},owner_surface=sm['owner_surface'])
            rows.append(row);evals[name,stage]=dest;manifests[name+'_'+stage]=m;print('SCORED',json.dumps(row),flush=True)
            dump(OUT/'progress.json',dict(stage=stage,variant=name,rows=len(rows),seconds=time.time()-started))
    pairs={}
    for stage in ['native','final']:
        for before,after in [('baseline','direct'),('baseline','e1'),('whole','e1'),('partial','e1'),('direct','e1')]:
            label=f'{before}_to_{after}_{stage}';out=OUT/'paired'/label
            if not (out/'paired_repair.json').exists():
                cli(['eval-repair-pair','--gt',gtpath,'--before',evals[before,stage]/'canonical_prediction.npz','--after',evals[after,stage]/'canonical_prediction.npz','--out',out],OUT/f'paired_{label}.log')
            pairs[label]=read(out/'paired_repair.json')
    pergt=[];pointchanges={}
    scope={int(x['raw_gt_id']):x for x in read(gtpath.parent/'gt_scope.json')['objects']}
    for stage in ['native','final']:
        overlaps={name:dict(np.load(path/'overlap_matrix.npz')) for (name,s),path in evals.items() if s==stage}
        gids=overlaps['baseline']['gt_ids']
        for j,gid in enumerate(gids):
            row=dict(scene=SCENE,group=GROUP,stage=stage,GT_ID=int(gid),semantic_class=scope[int(gid)]['semantic_class'])
            for name,z in overlaps.items():
                np.testing.assert_array_equal(z['gt_ids'],gids);row[name]=float(z['iou'][:,j].max(initial=0))
            for name in ['baseline','whole','partial','direct']:row['e1_minus_'+name]=row['e1']-row[name]
            pergt.append(row)
        arrays={name:dict(np.load(m['source_surface'])) for key,m in manifests.items() for name in [key.rsplit('_',1)[0]] if key.endswith('_'+stage)}
        for before,after in [('baseline','direct'),('baseline','e1'),('whole','e1'),('partial','e1'),('direct','e1')]:
            a,b=arrays[before]['instance_id'],arrays[after]['instance_id']
            pointchanges[f'{before}_to_{after}_{stage}']=dict(assigned_to_unassigned=int(((a>0)&(b<=0)).sum()),
                unassigned_to_assigned=int(((a<=0)&(b>0)).sum()),assigned_id_changed=int(((a>0)&(b>0)&(a!=b)).sum()))
    dump(OUT/'per_GT.json',pergt)
    result=dict(status='PASS',group=GROUP,scene=SCENE,selected=COUNT,GT_count=len(gids),protocol='object_observed_repair / revision 2',frozen=False,
        rows=rows,same_geometry_GT_scope_correspondence_and_scorer=True,point_changes=pointchanges,
        source_locks={k:{'source':m['source_surface'],'surface_sha256':m['source_surface_sha256'],**{f:m[f] for f in LOCKS}} for k,m in manifests.items()},
        mapping=read(R/'mapping_complete.json'),seconds=time.time()-started)
    dump(OUT/'comparison.json',result);dump(R/'evaluation_complete.json',dict(status='PASS',seconds=time.time()-started,rows=len(rows)))
    verify_sources();print('EVALUATION_COMPLETE',SCENE,GROUP,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:dump(R/'evaluation_failure.json',dict(traceback=traceback.format_exc()));raise
