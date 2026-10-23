"""Per-step frame input fence for mapping continuation."""
import os,sys,json,re
settings_path=os.environ.get('CVPR_ONLINE_GUARD')
current_cutoff=-1
if settings_path:
 with open(settings_path) as stream:settings=json.load(stream)
 current_cutoff=settings['initial_cutoff']
 root=settings['root'];allowed=set(settings['allowed_source_files'])
 caches=settings.get('allowed_prefix_cache_roots',[])
 input_roots=settings['restricted_input_roots'];result_roots=settings['restricted_result_roots']
 audit=root+'/io_audit';os.makedirs(audit,exist_ok=True)
 fd=os.open(audit+'/'+str(os.getpid())+'.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
 handling=False
 def beneath(path,parent):return path==parent or path.startswith(parent+'/')
 def guard(event,args):
  global handling
  if event!='open' or handling or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
  handling=True
  try:
   requested=os.fsdecode(args[0]);path=os.path.realpath(requested)
   if beneath(path,audit):return
   tracked=beneath(path,root) or path in allowed or any(beneath(path,x) for x in caches)
   reason=None
   if '/ground_truth/' in path or os.path.basename(path) in ['gt.npz','info_semantic.json','semantic.ply']:reason='GT_forbidden'
   if any(beneath(path,x) for x in input_roots) and path not in allowed:reason='outside_input_allowlist'
   if any(beneath(path,x) for x in result_roots) and not tracked:reason='foreign_result'
   if tracked:
    match=re.search(r'(?<![A-Za-z0-9])(?:frame|depth|f)(\d{6})(?!\d)',os.path.basename(path))
    if match and int(match.group(1))>current_cutoff:reason='future_frame'
   if tracked or reason:
    os.write(fd,(json.dumps(dict(path=path,requested=requested,cutoff=current_cutoff,blocked=bool(reason),reason=reason,mode=str(args[1] if len(args)>1 else None)))+'\n').encode())
   if reason:raise PermissionError('ONLINE_INPUT_FENCE: '+reason+': '+requested)
  finally:handling=False
 sys.addaudithook(guard)
