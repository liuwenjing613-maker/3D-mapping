from native_runtime import container
import subprocess,sys
sys.exit(subprocess.call(container(['/usr/bin/python3','-B','/experiment/native_geometry.py',sys.argv[1],'full'])))
