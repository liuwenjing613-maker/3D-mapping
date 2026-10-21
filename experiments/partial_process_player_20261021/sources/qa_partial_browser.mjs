import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
const require=createRequire(import.meta.url);
const {chromium}=require('C:/Users/liuwenjing/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve('results/partial逐帧检查器_20261021');
const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true});
const page=await browser.newPage({viewport:{width:1680,height:1120},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
const result={status:'RUNNING',runs:[],errors};
function assert(x,m){if(!x)throw Error(m);}
async function ready(){await page.waitForFunction(()=>window.__PARTIAL__?.engine&&!window.__PARTIAL__.loading&&document.querySelector('#app').hidden===false,{},{timeout:120000});}
try{
 for(const run of process.argv.slice(2).length?process.argv.slice(2):['manual_room1','manual_room0','automatic_room1','automatic_room0']){
  await page.goto('http://127.0.0.1:8917/?run='+run,{waitUntil:'domcontentloaded',timeout:120000});await ready();
  const row=await page.evaluate(()=>{const p=window.__PARTIAL__;return {run:p.meta.variant,events:p.meta.events.length,frame:p.fi*5,point:p.selected,rows:p.rows.length,title:document.querySelector('#frameTitle').textContent};});
  assert(row.run===run,'Wrong run');assert(row.events===1200,'Wrong event count');
  const proof=await page.evaluate(async()=>{
   const p=window.__PARTIAL__,i=p.fi,point=p.selected;
   const snap=()=>({label:p.engine.label[point],state:p.engine.state[point],votes:[...p.engine.votes(point).current],rows:p.rows.map(r=>[r.p,r.b,r.c,r.kind])});
   const before=JSON.stringify(snap());await p.go(Math.max(0,i-1));await p.go(i);return {roundtripExact:before===JSON.stringify(snap()),lockedPoint:point===p.selected};
  });assert(proof.roundtripExact,'Back-forward mismatch');assert(proof.lockedPoint,'Point changed');
  if(run==='manual_room1'){
   await page.selectOption('#region','9');await page.click('#fit');
   const stat=await page.evaluate(()=>({point:window.__PARTIAL__.selected,stats:document.querySelector('#stats').innerText,scope:document.querySelector('#scopeNote').innerText}));row.case=stat;
   await page.screenshot({path:path.join(root,'枕头_逐帧检查.png'),fullPage:true});
   await page.selectOption('#rightLayer','adopted');await page.screenshot({path:path.join(root,'枕头_实际采用Mask.png'),fullPage:true});await page.selectOption('#rightLayer','changes');
  }
  if(run==='manual_room0'){await page.selectOption('#region','85');await page.click('#fit');await page.screenshot({path:path.join(root,'书本_冲突弃权.png'),fullPage:true});await page.getByRole('button',{name:'有效换票 · 515',exact:true}).click();await ready();await page.screenshot({path:path.join(root,'书本_逐帧检查.png'),fullPage:true});}
  if(run==='automatic_room0'){await page.getByRole('button',{name:'插座GT56 · 1475',exact:true}).click();await ready();await page.screenshot({path:path.join(root,'插座_逐帧检查.png'),fullPage:true});}
  for(let stage=1;stage<3;stage++){await page.click(`[data-stage="${stage}"]`);await ready();assert(await page.evaluate(()=>window.__PARTIAL__.stage)===stage,'Stage switching failed');}
  await page.click('[data-stage="0"]');await ready();await page.locator('body').click({position:{x:20,y:20}});const old=await page.evaluate(()=>window.__PARTIAL__.fi);await page.keyboard.press('ArrowRight');await ready();assert(await page.evaluate(()=>window.__PARTIAL__.fi)===Math.min(old+1,399),'Keyboard forward failed');await page.keyboard.press('ArrowLeft');await ready();assert(await page.evaluate(()=>window.__PARTIAL__.fi)===old,'Keyboard reverse failed');
  await page.click('#mark');assert(await page.locator('.saved').count()>0,'Bookmark missing');
  await page.click('#setA');await page.click('#next');await ready();await page.click('#setB');await page.check('#loop');const bounds=await page.locator('#loopText').textContent();await page.selectOption('#speed','160');await page.click('#play');await page.waitForTimeout(900);await page.evaluate(()=>window.__PARTIAL__.pause());await ready();const loopFrame=await page.evaluate(()=>window.__PARTIAL__.fi*5);const [aa,bb]=bounds.split('–').map(Number);assert(loopFrame>=aa&&loopFrame<=bb,'Loop escaped interval');await page.uncheck('#loop');
  const beforeView=await page.evaluate(()=>window.__PARTIAL__.view);const box=await page.locator('#right').boundingBox();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.wheel(0,-200);await page.waitForTimeout(100);assert((await page.evaluate(()=>window.__PARTIAL__.view.scale))>beforeView.scale,'Zoom did not work');
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert(!overflow,'Horizontal overflow');
  result.runs.push({...row,...proof,stageSwitch:true,keyboard:true,bookmark:true,ABLoop:true,synchronizedZoom:true,horizontalOverflow:false});console.log(JSON.stringify(result.runs.at(-1)));
 }
 assert(errors.length===0,'Browser console errors: '+errors.join('; '));result.status='PASS';
}catch(e){result.status='FAIL';result.failure=e.stack;console.error(e);}
finally{fs.writeFileSync(path.join(root,'qa_browser.json'),JSON.stringify(result,null,2));await browser.close();}
if(result.status!=='PASS')process.exitCode=1;
