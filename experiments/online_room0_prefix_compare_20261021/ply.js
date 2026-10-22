export function parsePLY(buffer){
 const preview=new TextDecoder().decode(new Uint8Array(buffer,0,Math.min(buffer.byteLength,4096)));
 const end=preview.indexOf('end_header\n');if(end<0||!preview.startsWith('ply\nformat binary_little_endian 1.0\n'))throw Error('PLY 格式不符合实际导出文件');
 const header=preview.slice(0,end),offset=end+11;
 const count=Number(header.match(/element vertex (\d+)/)?.[1]);
 const expected=['float x','float y','float z','uchar red','uchar green','uchar blue','int instance_id','uchar state','int surface_index'];
 const properties=header.split('\n').filter(x=>x.startsWith('property ')).map(x=>x.slice(9));
 if(JSON.stringify(properties)!==JSON.stringify(expected)||!Number.isInteger(count)||buffer.byteLength!==offset+count*24)throw Error('PLY 属性或字节数不匹配');
 const view=new DataView(buffer),positions=new Float32Array(count*3),rgb=new Uint8Array(count*3),ids=new Float32Array(count),states=new Float32Array(count);
 for(let i=0,p=offset;i<count;i++,p+=24){
  for(let j=0;j<3;j++){positions[3*i+j]=view.getFloat32(p+4*j,true);rgb[3*i+j]=view.getUint8(p+12+j);}
  ids[i]=view.getInt32(p+15,true);states[i]=view.getUint8(p+19);
  if(view.getInt32(p+20,true)!==i)throw Error('表面点索引不连续');
 }
 return {count,positions,rgb,ids,states};
}
export function instanceColors(ids,states){
 const colors=new Uint8Array(ids.length*3),cache=new Map();
 for(let i=0;i<ids.length;i++){
  const id=ids[i];let c;
  if(id<=0)c=states[i]===1?[255,255,0]:states[i]===3?[255,0,0]:[0,0,0];
  else{
   c=cache.get(id);
   if(!c){const h=((id*.61803398875)%1)*6,s=.67,v=.94,k=Math.floor(h),f=h-k,p=v*(1-s),q=v*(1-s*f),t=v*(1-s*(1-f));
    c=[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][k%6].map(x=>Math.floor(x*255));cache.set(id,c);
   }
  }
  colors.set(c,i*3);
 }
 return colors;
}