from pathlib import Path
import json,os,sys
C=Path(__file__).resolve().parent
GROUP=os.environ['POINT_GROUP']; SCENE=os.environ['POINT_SCENE']
CFG=json.loads((C/'config.json').read_text())[GROUP][SCENE]
SOURCE=Path(CFG['source']); SOURCE_CODE=Path(CFG['source_code']); PARTIAL=Path(CFG['partial'])
COUNT=CFG['selected']; R=Path('/data/chenkejun/CVPR/results/frame_point_replacement_20261021')/GROUP/SCENE
MAIN=Path('/home/chenkejun/CVPR/worktrees/v3-r2-main-20261009')
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources,LockedModules,load_script
verify_sources()
F=MAIN/'instance_repair/frozen_e1_20261012'/('room2' if SCENE=='room2' else 'room0')
mapping=json.loads((F/'module_map.json').read_text()); mapping['run_history_core']=str(SOURCE_CODE/'run_history_core.py')
sys.meta_path.insert(0,LockedModules(F,mapping))
sys.path.insert(0,str(C))
