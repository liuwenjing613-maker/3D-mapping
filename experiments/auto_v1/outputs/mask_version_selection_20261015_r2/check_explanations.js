const fs=require('fs'),vm=require('vm'),path=require('path');
const dir=__dirname,code=fs.readFileSync(path.join(dir,'selection_ui.js'),'utf8');
const context={};vm.createContext(context);
vm.runInContext(code.slice(code.indexOf('const reasonNames='),code.indexOf('let reviewQuality='))+'\n'+code.slice(code.indexOf('function nodeName('),code.indexOf('function inspectFromReason('))+'\nthis.explain=reasonForNode;',context);
const data=JSON.parse(fs.readFileSync(path.join(dir,'selection.json'),'utf8'));
let count=0;
for(const c of Object.values(data.crops)){
 const groups=new Map(c.groups.map(g=>[g.id,g]));
 for(const n of c.nodes){
  const row=c.models[n.model].labels[n.label];
  if(row.status!=='deferred')continue;
  const explanation=context.explain(c,n,groups.get(row.group));
  if(!explanation.reason||!explanation.evidence||/undefined|NaN/.test(explanation.reason+explanation.evidence))throw Error(c.ROI+' '+n.key+' missing explanation');
  count++;
 }
}
if(count!==data.stats.deferred_occurrences)throw Error('incomplete deferred coverage');
console.log('PASS: all '+count+' deferred Mask IDs have concrete reasons and evidence.');
