from pathlib import Path
import os,sys,json,subprocess,traceback
C=Path(__file__).resolve().parent;O=Path('/data/chenkejun/CVPR/results')/C.name
s=sys.argv[1];status=O/(s+'_status.json')
def run(stage,script,*args):
 status.write_text(json.dumps(dict(status='RUNNING',scene=s,stage=stage,amendment='invalid_normals_v2',pid=os.getpid())))
 with (O/(s+'_'+stage+'_resume.log')).open('a') as f:subprocess.run([sys.executable,'-B','-u',str(C/script),*args],cwd=C,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 if not (O/'main'/s/'200/A/mapping/mapping_complete.json').exists():run('strict_mapping','run_mapping_stage_A.py',s,'200')
 run('final_B','finalize_baseline.py',s)
 run('original_evidence','build_original_dbscan_evidence.py',s)
 run('score_baseline','evaluate_scene.py',s,'no_repair')
 run('DBSCAN','run_dbscan.py',s)
 status.write_text(json.dumps(dict(status='PASS',scene=s,amendment='invalid_normals_v2')))
except Exception:
 status.write_text(json.dumps(dict(status='FAILED',scene=s,traceback=traceback.format_exc())));raise
