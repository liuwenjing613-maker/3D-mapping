"""Separate same-surface control; each side keeps its independently selected real seeds."""
from pathlib import Path
import ast,hashlib,json

HERE=Path(__file__).resolve().parent
OUT=Path('/data/chenkejun/CVPR/results/online_checkpoint_closed_loop_20261023')
MAPS=Path('/data/chenkejun/CVPR/results/online_checkpoint_surface_20261023')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def link(src,dst):
    assert src.exists(),src
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())

def main():
    assert read(MAPS/'common_surface_comparison_200_A.json')['old_new']['status']=='PASS'
    audit=[]
    for mode in ['reference','checkpoint']:
        src=HERE/mode;code=HERE/('fixed_'+mode)
        actual=OUT/mode;root=OUT/('fixed_'+mode)
        common=MAPS/('common_surface_'+mode+'_200_A')
        assert read(actual/'candidates/online_qa.json')['status']=='PASS'
        assert read(actual/'repair/commit_complete.json')['status']=='PASS'
        assert not root.exists() and not code.exists()
        root.mkdir();code.mkdir()
        files=[src/n for n in ['common_continue.py','bootstrap_continue.py','scene_continue.py','selection_runtime.py',
            'selection_fence.py','track_prepare.py','track_queue.py','track_queue_with_cache.py','track_gate.py',
            'track_batch.py','track_pure_gate.py','track_case.py','repair_continue.py','commit_continue.py']]
        files+=list((src/'selection_guard').glob('*.py'))
        files+=[p for p in (src/'tracking_code/200/A').iterdir() if p.is_file()]
        for p in files:
            rel=p.relative_to(src);text=p.read_text().replace(str(src),str(code)).replace(str(actual),str(root))
            if p.suffix=='.py':ast.parse(text)
            dest=code/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
        assert sha(code/'track_case.py')==sha(src/'track_case.py')
        mapping=root/'mapping';mapping.mkdir()
        link(common/'baseline',mapping/'baseline')
        for name in ['config.json','inputs','association','prior_commit.json','mapping_complete.json']:
            link(actual/'mapping'/name,mapping/name)
        for name in ['surface.ply','review_surface_graph_n02_p08.npz']:
            link(common/'geometry'/name,mapping/'geometry'/name)
        link(actual/'mapping/geometry/tsdf_grid.npz',mapping/'geometry/tsdf_grid.npz')
        link(actual/'candidates',root/'candidates')
        link(actual/'tracking/inputs',root/'tracking/inputs')
        cfg=read(actual/'tracking/track_config.json')
        write(root/'tracking/track_config.json',cfg)
        write(root/'inputs_complete.json',read(actual/'inputs_complete.json'))
        guard=read(actual/'input_guard.json');guard['root']=str(root)
        guard['allowed_prefix_cache_roots']=sorted(set(guard['allowed_prefix_cache_roots']+[str(actual),str(root),str(common)]))
        guard['allowed_source_files']=sorted(set(guard['allowed_source_files']+[str((mapping/'geometry/surface.ply').resolve())]))
        guard['read_only_result_roots']=sorted(set(guard.get('read_only_result_roots',[])+[str(actual),str(common)]))
        write(root/'input_guard.json',guard)
        provenance=dict(status='READY',verification_only=True,original_run=str(actual),
            selected_seeds_source=str(actual/'candidates/all_edge_20261021/v2/seed_manifest.json'),
            selected_seeds_sha256=sha(actual/'candidates/all_edge_20261021/v2/seed_manifest.json'),
            common_surface_from_own_unchanged_voting_replay=str(common),
            surface_evidence_sha256=sha(common/'baseline/surface_evidence.npz'),
            no_candidate_unification=True,no_vertex_sorting=True,no_online_algorithm_change=True)
        write(root/'fixed_surface_control.json',provenance);audit.append(provenance)
    write(OUT/'fixed_control_setup.json',dict(status='PASS',controls=audit))
    print('FIXED_CONTROL_PREPARED',flush=True)
if __name__=='__main__':main()
