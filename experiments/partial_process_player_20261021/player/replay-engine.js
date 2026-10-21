// Pure playback logic. All vote deltas come from verified saved transactions.
export function decodeBundle(buffer){
  const size=new DataView(buffer).getUint32(0,true);
  const schema=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,size)));
  const types={'<u4':Uint32Array,'<i4':Int32Array,'<f4':Float32Array,'|u1':Uint8Array,'<u2':Uint16Array,'<i2':Int16Array,'|i1':Int8Array};
  const arrays={};
  for(const [name,s] of Object.entries(schema)){
    const Type=types[s.dtype];if(!Type)throw new Error(`Unsupported dtype: ${s.dtype}`);
    arrays[name]=new Type(buffer,4+size+s.offset,s.length);
  }
  return arrays;
}
export function lowerBound(a,value,lo=0,hi=a.length){while(lo<hi){const m=(lo+hi)>>>1;if(a[m]<value)lo=m+1;else hi=m;}return lo;}
function idsAt(a,off,event,point,base){
  const lo=lowerBound(a,point*base,off[event],off[event+1]);
  const hi=lowerBound(a,(point+1)*base,lo,off[event+1]);
  const ids=[];for(let i=lo;i<hi;i++)ids.push(a[i]%base);return ids;
}
export function voteState(counts){
  const ordered=[...counts].filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]||a[0]-b[0]);
  const total=ordered.reduce((s,x)=>s+x[1],0),top=ordered[0]||[-1,0],second=ordered[1]||[-1,0],ratio=total?top[1]/total:0;
  const state=!total?0:top[1]>=2&&ratio>=.67?2:second[1]>0&&ratio<.67?3:1;
  return {ordered,total,top:top[0],votes:top[1],second:second[0],ratio,state,label:state===2?top[0]:-1};
}
export class ReplayEngine{
  constructor(arrays,meta){
    if(meta.keyEncoding==='per-event-positive-delta-v1'&&!arrays._decoded){
      for(const [key,off] of [['removed','r'],['added','a'],['changedPoint','p']]){
        const values=arrays[key],offsets=arrays['offset_'+off];
        for(let e=0;e<meta.events.length;e++){let sum=0;for(let i=offsets[e];i<offsets[e+1];i++){sum+=values[i];values[i]=sum;}}
      }
      let sum=0;for(let i=0;i<arrays.voteKey.length;i++){sum+=arrays.voteKey[i];arrays.voteKey[i]=sum;}arrays._decoded=true;
    }
    this.a=arrays;this.meta=meta;this.base=meta.base;this.label=arrays.initialLabel.slice();this.state=arrays.initialState.slice();this.executed=0;this.cache=new Map();
  }
  jump(count){
    const a=this.a;count=Math.max(0,Math.min(this.meta.events.length,count));
    while(this.executed<count){const e=this.executed++;for(let i=a.offset_p[e];i<a.offset_p[e+1];i++){this.label[a.changedPoint[i]]=a.newLabel[i];this.state[a.changedPoint[i]]=a.newState[i];}}
    while(this.executed>count){const e=--this.executed;for(let i=a.offset_p[e];i<a.offset_p[e+1];i++){this.label[a.changedPoint[i]]=a.oldLabel[i];this.state[a.changedPoint[i]]=a.oldState[i];}}
  }
  initialVotes(point){
    const a=this.a,lo=lowerBound(a.voteKey,point*this.base),hi=lowerBound(a.voteKey,(point+1)*this.base,lo),counts=new Map();
    for(let i=lo;i<hi;i++)counts.set(a.voteKey[i]%this.base,a.voteCount[i]);return counts;
  }
  operations(point){
    if(this.cache.has(point))return this.cache.get(point);
    const a=this.a,ops=[];
    for(let e=0;e<this.meta.events.length;e++){
      const removed=idsAt(a.removed,a.offset_r,e,point,this.base),added=idsAt(a.added,a.offset_a,e,point,this.base);
      if(removed.length||added.length)ops.push({event:e,removed,added});
    }
    if(this.cache.size>60)this.cache.delete(this.cache.keys().next().value);this.cache.set(point,ops);return ops;
  }
  votes(point,count=this.executed){
    const initial=this.initialVotes(point),current=new Map(initial),ops=this.operations(point);
    for(const op of ops){if(op.event>=count)break;for(const id of op.removed)current.set(id,(current.get(id)||0)-1);for(const id of op.added)current.set(id,(current.get(id)||0)+1);}
    for(const [id,v] of current){if(v<0)throw new Error(`Negative vote at point ${point}, ID ${id}`);if(v===0)current.delete(id);}
    return {initial,current,operations:ops,summary:voteState(current)};
  }
  candidates(point,event){
    const d=this.details;if(event<0||!d||event<d.start||event>=d.end)return null;
    const i=event-d.start,before=idsAt(d.a.candidateBefore,d.a.offset_b,i,point,this.base),after=idsAt(d.a.candidateAfter,d.a.offset_c,i,point,this.base);
    if(!before.length&&!after.length)return null;
    const op=this.operations(point).find(x=>x.event===event);
    return {before,after,removed:op?.removed||[],added:op?.added||[]};
  }
}
