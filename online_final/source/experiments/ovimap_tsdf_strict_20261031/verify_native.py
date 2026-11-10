from pathlib import Path
import numpy as np,json,hashlib,re
O=Path('/data/chenkejun/CVPR/results/ovimap_tsdf_strict_20261031')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=5);args=p.parse_args();count=args.count;fid=(count-1)*5
 a=O/('native/room0_%d_trace'%count);b=O/('native/room0_%d_terminal'%count);c=O/('native/room0_%d_original'%count)
 for p in [a,b]:assert (p/'readout_complete.json').exists(),str(p)
 with np.load(a/'tsdf_final.npz') as x,np.load(b/'tsdf_final.npz') as y:
  for k in x.files:np.testing.assert_array_equal(x[k],y[k])
 with np.load(a/('ovi/f%06d.npz'%fid)) as x,np.load(b/('ovi/f%06d.npz'%fid)) as y:
  for k in x.files:np.testing.assert_array_equal(x[k],y[k])
 # Official PLY export: byte equality includes all vertices, colors and faces.
 meshes=[p/('cropformer_inst/instance_mesh_%d.ply'%count) for p in [a,b,c]]
 assert all(p.exists() for p in meshes),str(meshes)
 hashes=[sha(p) for p in meshes]
 assert len(set(hashes))==1,hashes
 # Read integer IDs vs native per-vertex palette export, preserving mesh entry order after sorting.
 import plyfile
 data=plyfile.PlyData.read(meshes[0])['vertex'].data
 names=data.dtype.names
 xyz=np.column_stack([data[k] for k in ('x','y','z')]).astype(np.float32)
 colors=np.column_stack([data[k] for k in ('red','green','blue')]).astype(np.int64)
 logs=''.join(p.read_text(errors='replace') for p in (a/'log').glob('*INFO*'))
 palette={}
 for ident,r,g,bb in re.findall(r'Instance:\s*(\d+)\s+Color:\s*\((\d+),(\d+),(\d+)\)',logs):
  key=(int(r),int(g),int(bb))
  assert key not in palette or palette[key]==int(ident);palette[key]=int(ident)
 assert palette
 # PLY serialization rounds coordinates, so compare nearest under its precision.
 from scipy.spatial import cKDTree
 with np.load(a/('ovi/f%06d.npz'%fid)) as z:
  distance,index=cKDTree(z['xyz_m']).query(xyz,workers=2)
  decoded=np.asarray([palette.get(tuple(col),-1) for col in colors],np.int32)
  expected=z['instance_id'][index];decoded[decoded<=0]=-1
  assert float(distance.max())<1e-5
  assert np.array_equal(decoded,expected),int(np.sum(decoded!=expected))
 report={'status':'PASS','frames':count,'tsdf_trace_vs_terminal_bit_exact':True,'official_mesh_trace_terminal_original_byte_exact':True,'official_vertex_palette_vs_integer_ids_exact':True,'mesh_sha256':hashes[0],'max_ply_rounding_error_m':float(distance.max())}
 (O/('native_smoke_validation_%d.json'%count)).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
