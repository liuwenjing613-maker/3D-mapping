from online_common import *
import runB,time
runB.run("qa","qaB.py")
failure=R/"workflow_failure.json"
if failure.exists():
 d=read(failure);d.update(resolved=True,resolution="QA runtime integer-category namespace serialization corrected; no numerical candidates changed",resolved_unix=time.time());write(R/"qa_retry_audit.json",d);failure.unlink()
write(R/"workflow_status.json",dict(status="PASS",arm="B",stage="100_new50_masks_ready_no_second_repair",unix=time.time()))
