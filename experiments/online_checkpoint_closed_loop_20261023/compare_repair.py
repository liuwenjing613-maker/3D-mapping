"""Audit repair arrays, allowed regions, transactions, and committed identity/pixel state."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import open3d as o3d
from PIL import Image

OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def order(a):return np.lexsort(tuple(a[:,i] for i in range(a.shape[1]-1,-1,-1)))

def compare(left,right,fixed=False):
    checks=[]
    def value(name,a,b):
        checks.append(dict(name=name,exact=a==b))
    def array(name,a,b,pointwise=False):
        exact=a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
        row=dict(name=name,exact=bool(exact),left_shape=list(a.shape),right_shape=list(b.shape))
        if a.shape==b.shape and not exact:
            delta=a!=b
            row['different_elements']=int(delta.sum())
            if a.ndim:
                changed=delta.reshape(len(a),-1).any(1)
                row['different_rows']=int(changed.sum())
                if pointwise:
                    row['nonduplicate_different_rows']=int((changed&unique_coordinate).sum())
            if a.dtype.kind in 'fiu' and a.size:row['max_abs_difference']=float(np.max(np.abs(a.astype(float)-b.astype(float))))
        checks.append(row)
    orders=[];inverse=[];attrs=[]
    for root in [left,right]:
        cloud=o3d.io.read_point_cloud(str(root/'mapping/geometry/surface.ply'))
        geometry=np.column_stack((np.asarray(cloud.points),np.asarray(cloud.normals),np.asarray(cloud.colors)))
        z=load(root/'mapping/baseline/surface_evidence.npz')
        native=load(root/'mapping/baseline/native/instance_surface.npz')['instance_id']
        final=load(root/'mapping/baseline/final/instance_surface.npz')['instance_id']
        records=[geometry]+[z[k][:,None] for k in sorted(z) if k not in ['xyz_m','rgb']]+[native[:,None],final[:,None]]
        idx=np.arange(len(geometry)) if fixed else order(np.column_stack(records))
        orders.append(idx);inv=np.empty(len(idx),np.int64);inv[idx]=np.arange(len(idx));inverse.append(inv);attrs.append(geometry)
    _,coordinate_index,coordinate_counts=np.unique(attrs[0][orders[0],:3],axis=0,return_inverse=True,return_counts=True)
    unique_coordinate=coordinate_counts[coordinate_index]==1
    array('geometry/one_global_alignment',attrs[0][orders[0]],attrs[1][orders[1]])
    bases=[read(p/'tracking/global_association.json')['instance_base'] for p in [left,right]]
    value('repair/instance_base',*bases)
    def keys(x,inv,base):return np.sort(inv[x//base]*base+x%base)
    def points(x,inv):return np.sort(inv[x])
    for stage in ['direct','core_all','whole_to_partial']:
        folder=Path('repair/partial')/stage/'room0'
        for rel in ['native/instance_surface.npz','final/instance_surface.npz','native/surface_evidence.npz']:
            a,b=[load(p/folder/rel) for p in [left,right]]
            value(str(folder/rel)+'/keys',sorted(a),sorted(b))
            for k in sorted(a.keys()&b.keys()):
                x,y=a[k],b[k];pointwise=k!='native_instance_ids'
                if pointwise:x,y=x[orders[0]],y[orders[1]]
                array(str(folder/rel)+'/'+k,x,y,pointwise)
        tables=[]
        for p,inv in zip([left,right],inverse):
            z=load(p/folder/'native/surface_instance_frame_votes.npz')
            t=np.column_stack((inv[z['surface_point_index']],z['instance_id'],z['frame_votes']))
            tables.append(t[order(t)])
        array(str(folder)+'/strict_votes',*tables)
        tables=[points(load(p/folder/'allowed_surface_ids.npz')['surface_point_index'],inv) for p,inv in zip([left,right],inverse)]
        array(str(folder)+'/allowed_surface_ids',*tables)
    for rel in ['repair/e1/region.npz','repair/e1/adaptive_cores.npz']:
        a,b=[load(p/rel) for p in [left,right]]
        value(rel+'/keys',sorted(a),sorted(b))
        for k in a.keys()&b.keys():
            if k=='surface_point_index':x,y=points(a[k],inverse[0]),points(b[k],inverse[1]);pointwise=False
            else:x,y=a[k][orders[0]],b[k][orders[1]];pointwise=True
            array(rel+'/'+k,x,y,pointwise)
    transaction_diff=[];transaction_checks=0
    for fid in range(0,996,5):
        rel=f'repair/partial/transactions/f{fid:06d}.npz'
        a,b=[load(p/rel) for p in [left,right]]
        assert sorted(a)==sorted(b)
        for k in a:
            x,y=keys(a[k],inverse[0],bases[0]),keys(b[k],inverse[1],bases[1]);transaction_checks+=1
            if not np.array_equal(x,y):transaction_diff.append(dict(frame=fid,field=k,left_size=len(x),right_size=len(y),symmetric_difference=int(len(np.setxor1d(x,y)))))
    checks.append(dict(name='repair/per_frame_transactions',exact=not transaction_diff,comparisons=transaction_checks,differences=transaction_diff))
    journals=[read(p/'repair/journal/manifest.json')['frames'] for p in [left,right]]
    pixel_diffs=[];pixel_files=0
    assert len(journals[0])==len(journals[1])==200
    for a,b in zip(*journals):
        assert a['frame_id']==b['frame_id'] and len(a['layer_files'])==len(b['layer_files'])
        for layer,(x,y) in enumerate(zip(a['layer_files'],b['layer_files'])):
            pa,pb=Path(x['path']),Path(y['path']);assert sha(pa)==x['sha256'] and sha(pb)==y['sha256'];pixel_files+=1
            if x['sha256']!=y['sha256']:
                aa,bb=np.asarray(Image.open(pa)),np.asarray(Image.open(pb))
                if not np.array_equal(aa,bb):pixel_diffs.append(dict(frame=a['frame_id'],layer=layer,different_pixels=int(np.count_nonzero(aa!=bb))))
    checks.append(dict(name='commit/history_pixel_layers',exact=not pixel_diffs,files=pixel_files,differences=pixel_diffs))
    commits=[read(p/'repair/commit_complete.json') for p in [left,right]]
    for p in commits:
        assert p['status']=='PASS'
        for key in ['checkpoint','identity_counts','source_pixel_support','observation_lineage','audit','journal_manifest','final_map','geometry']:
            assert sha(Path(p[key]))==p[key+'_sha256'],key
        assert read(Path(p['audit']))['status']=='PASS'
    for key in ['processed_frames','last_raw_frame','map_version','next_instance_id']:
        value('commit/'+key,commits[0][key],commits[1][key])
    for key in ['checkpoint','identity_counts','source_pixel_support']:
        a,b=[load(Path(p[key])) for p in commits]
        value('commit/'+key+'/keys',sorted(a),sorted(b))
        for k in a.keys()&b.keys():array('commit/'+key+'/'+k,a[k],b[k])
    lineage=[read(Path(p['observation_lineage']))['entries'] for p in commits]
    normalized=[];input_sources=[]
    for rows,root in zip(lineage,[left,right]):
        inputs=(root/'mapping/inputs').resolve();assert inputs.name=='inputs'
        sources={}
        for row in rows:
            path=Path(row['sampling_source_mask'])
            if path.is_relative_to(inputs):
                relative=str(path.relative_to(inputs))
                if relative not in sources:sources[relative]=sha(path)
        input_sources.append(sources)
        normalized.append(json.loads(json.dumps(rows).replace(str(root/'mapping'),'<MAPPING>').replace(str((root/'mapping').resolve()),'<MAPPING>').replace(str(inputs.parent),'<MAPPING>').replace(str(root),'<RUN>')))
    # Fixed controls link mapping/inputs individually. Verify the actual source bytes before normalizing that resolved prefix.
    value('commit/sampling_source_mapping_input_bytes',*input_sources)
    # Explicit normalization removes run-dependent path prefixes only; hashes and IDs remain compared.
    value('commit/observation_lineage_entries_path_normalized',*normalized)
    if fixed:
        for folder in ['e1','modified_repair','partial','commit_candidate']:
            files=[sorted(p.relative_to(root/'repair') for p in (root/'repair'/folder).rglob('*.npz')) for root in [left,right]]
            value('fixed_raw/'+folder+'/files',*files)
            for rel in sorted(set(files[0])&set(files[1])):
                a,b=[load(root/'repair'/rel) for root in [left,right]]
                value('fixed_raw/'+str(rel)+'/keys',sorted(a),sorted(b))
                for k in a.keys()&b.keys():array('fixed_raw/'+str(rel)+'/'+k,a[k],b[k])
    return dict(status='PASS' if all(r['exact'] for r in checks) else 'DIFFERENT',left=str(left),right=str(right),
                fixed_identical_surface=fixed,zero_float_tolerance=True,checks=checks)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--left',default='reference');parser.add_argument('--right',default='checkpoint');parser.add_argument('--name',default='repair_comparison');parser.add_argument('--fixed',action='store_true')
    args=parser.parse_args();result=compare(OUT/args.left,OUT/args.right,args.fixed)
    (OUT/(args.name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],comparisons=len(result['checks']),different=[{k:v for k,v in row.items() if k!='differences'} for row in result['checks'] if not row['exact']]),indent=2))
