import fs from 'node:fs';import path from 'node:path';import zlib from 'node:zlib';import {pathToFileURL} from 'node:url';
const root=path.resolve('results/partial逐帧检查器_20261021');
const {decodeBundle,ReplayEngine}=await import(pathToFileURL(path.join(root,'replay-engine.js')));
function unpack(p){const b=zlib.gunzipSync(fs.readFileSync(p));return decodeBundle(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));}
function assert(x,m){if(!x)throw Error(m);}
const results=[];
for(const run of ['automatic_room0','automatic_room1','manual_room0','manual_room1']){
 const dir=path.join(root,'assets',run),m=JSON.parse(fs.readFileSync(path.join(dir,'data.json'))),a=unpack(path.join(dir,'replay.bin.gz')),e=new ReplayEngine(a,m),r={run,stageChecks:[],frameChecks:0,candidateChecks:0,pointLedgerChecks:0};
 assert(m.events.length===1200,'Missing frames');
 for(let s=1;s<=3;s++){e.jump(s*400);for(let i=0;i<e.label.length;i++){assert(e.label[i]===a['native'+s][i],run+' native map');assert(e.state[i]===a['state'+s][i],run+' state');}r.stageChecks.push({stage:s,allPointsExact:true});}
 e.jump(0);for(let i=0;i<e.label.length;i++){assert(e.label[i]===a.initialLabel[i],run+' reverse labels');assert(e.state[i]===a.initialState[i],run+' reverse states');}
 for(let n=0;n<1200;n++){
  const d=unpack(path.join(dir,'detail',String(n).padStart(4,'0')+'.bin.gz')),ev=m.events[n],points=new Set();assert(d.point.length===ev.points,'Point count');assert(d.uv.length===d.point.length*2,'UV count');assert(d.source.length===ev.sourcePixels*5,'Source count');
  for(let i=0;i<d.point.length;i+=Math.max(1,Math.floor(d.point.length/7)))points.add(d.point[i]);
  const candidates=(keys,p)=>{let lo=0,hi=keys.length;while(lo<hi){const mid=(lo+hi)>>>1;if(keys[mid]<p*m.base)lo=mid+1;else hi=mid;}const result=[];while(lo<keys.length&&Math.floor(keys[lo]/m.base)===p)result.push(keys[lo++]%m.base);return result;};
  const delta=(keys,offset,p)=>{for(let i=offset[n];i<offset[n+1];i++){if(Math.floor(keys[i]/m.base)===p)return keys[i]%m.base;if(Math.floor(keys[i]/m.base)>p)break;}return 0;};
  for(const p of points){const b=candidates(d.before,p),c=candidates(d.after,p),old=b.length===1?b[0]:0,now=c.length===1?c[0]:0;assert(delta(a.removed,a.offset_r,p)===(old&&old!==now?old:0),'Effective removal mismatch');assert(delta(a.added,a.offset_a,p)===(now&&now!==old?now:0),'Effective addition mismatch');r.candidateChecks++;}
  assert(a.offset_r[n+1]-a.offset_r[n]===ev.removed,'Removed count');assert(a.offset_a[n+1]-a.offset_a[n]===ev.added,'Added count');r.frameChecks++;
  if(n%79===0&&points.size){e.jump(n+1);for(const p of points){const v=e.votes(p).summary;assert(v.label===e.label[p],'Point ledger label');assert(v.state===e.state[p],'Point ledger state');r.pointLedgerChecks++;}}
 }
 for(let i=0;i<400;i++){const f='f'+String(i*5).padStart(6,'0');for(const kind of ['original','adopted','retired'])assert(fs.existsSync(path.join(dir,kind,f+'.png')),'Missing mask');assert(fs.existsSync(path.join(root,'assets',m.scene,'rgb',f+'.jpg')),'Missing RGB');}
 r.fullReverseExact=true;r.sourceFrames=400;r.allImageFilesPresent=true;r.status='PASS';results.push(r);console.log(JSON.stringify(r));
}
fs.writeFileSync(path.join(root,'qa_engine.json'),JSON.stringify({status:'PASS',results},null,2));
