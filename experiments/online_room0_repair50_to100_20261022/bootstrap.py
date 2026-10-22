from common import *
import sys
sys.path.insert(0,str(MAIN))
from instance_repair.cli import verify_sources,LockedModules,load_script
verify_sources()
F=MAIN/"instance_repair/frozen_e1_20261012/room0"
mapping=read(F/"module_map.json")
for n in ("run_history_core","run_case","objectwise_ops","gate_v1","strict_vote_ops"):
 if (C/(n+".py")).exists():mapping[n]=str(C/(n+".py"))
sys.meta_path.insert(0,LockedModules(F,mapping))
sys.path.insert(0,str(C))
import run_repairs as r
from prefix_scene import PrefixScene
r.Scene=PrefixScene
