from pathlib import Path
import tarfile,hashlib,json
L=Path(__file__).resolve().parent
files=['package_ply.py','unpack_review.py','diagnose_final.py','install_page.py','viewer.html','viewer.js','adapt_viewer.py','write_final_report.py','archive_final.py','package_delivery_code.py','install_delivery_code.py','FINAL_REPORT.txt','delivery_verification.json','browser_verification.json','timing_delivery.json','office4_B400_GT19.png','office2_B400_GT89.png','CONTINUATION.txt']
manifest={}
with tarfile.open(L/'delivery_code.tar.gz','w:gz') as tar:
 for name in files:
  p=L/name;assert p.is_file();tar.add(p,arcname=name);manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(L/'delivery_code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FILES',len(files),'BYTES',(L/'delivery_code.tar.gz').stat().st_size)
