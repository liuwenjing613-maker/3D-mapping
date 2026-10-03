const {createRequire}=require('module');
const {pathToFileURL}=require('url');
const path=require('path');
const fs=require('fs');
const deps=createRequire('C:/Users/liuwenjing/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const {chromium}=deps('playwright');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1024,height:1000},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.resolve(__dirname,'roi_preview_standalone.html')).href);
 const frame=page.frameLocator('iframe');
 await frame.locator('#roi-choice').waitFor();
 await frame.locator('#roi-states').evaluate(im=>im.decode());
 const initial=await frame.locator('#roi-summary').textContent();
 if(!initial.includes('ROI-0001'))throw Error('Wrong initial selection');
 const optionCount=await frame.locator('#roi-choice option').count();
 if(optionCount!==13)throw Error('Incomplete representative preview');
 await frame.locator('#roi-choice').selectOption('35');
 await frame.locator('#roi-states').evaluate(im=>im.decode());
 const selected=await frame.locator('#roi-summary').textContent();
 if(!selected.includes('ROI-0035')||!selected.includes('ROI 28'))throw Error('ROI selection failed to update');
 if(!(await frame.locator('#roi-relation').textContent()).includes('1.54m'))throw Error('Spatial relation annotation missing');
 if(!(await frame.locator('#roi-votes').textContent()).includes('其他 24'))throw Error('Multi-ID vote evidence missing');
 await frame.locator('#roi-cloud-mode').selectOption('instances');
 await frame.locator('#roi-cloud-view').selectOption('front');
 await frame.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
 await page.waitForTimeout(200);
 // The standalone test wrapper fixes iframe height to the viewport. The app
 // sizes it from content; match that behaviour for a complete mobile screenshot.
 const contentHeight=await frame.locator('#p1a1-room0-roi-review').evaluate(r=>Math.ceil(r.getBoundingClientRect().height)+32);
 await page.locator('iframe').evaluate((e,h)=>e.style.height=h+'px',contentHeight);
 const rendered=await frame.locator('#roi-cloud').evaluate(c=>{
  const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;
  for(let i=3;i<a.length;i+=4)if(a[i])n++;return n;
 });
 if(rendered<50)throw Error('Empty 3D canvas');
 await page.screenshot({path:path.resolve(__dirname,'roi_preview_desktop.png'),fullPage:true});
 await page.setViewportSize({width:360,height:1000});
 await frame.locator('#roi-choice').selectOption('226');
 await frame.locator('#roi-states').evaluate(im=>im.decode());
 await frame.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
 await page.waitForTimeout(200);
 const mobile=await frame.locator('#p1a1-room0-roi-review').evaluate(r=>({scroll:r.scrollWidth,width:r.clientWidth}));
 if(mobile.scroll>mobile.width+2)throw Error(`Mobile horizontal overflow: ${JSON.stringify(mobile)}`);
 const mobileContentHeight=await frame.locator('#p1a1-room0-roi-review').evaluate(r=>Math.ceil(r.getBoundingClientRect().height)+32);
 await page.setViewportSize({width:360,height:Math.max(2400,mobileContentHeight+64)});
 await page.locator('iframe').evaluate((e,h)=>e.style.height=h+'px',mobileContentHeight);
 await page.screenshot({path:path.resolve(__dirname,'roi_preview_mobile.png'),fullPage:true});
 for(const rank of [1,23,35,38,48,49,58,61,63,67,73,74,226]){
  await frame.locator('#roi-choice').selectOption(String(rank));
  await frame.locator('img').evaluateAll(images=>Promise.all(images.map(im=>im.decode())));
  if(!(await frame.locator('#roi-fact').textContent()).length)throw Error('Missing case annotation '+rank);
 }
 await frame.locator('#roi-choice').selectOption('73');
 if((await frame.locator('#roi-aux img').count())!==0)throw Error('Non-conflict example received competing masks');
 if(!(await frame.locator('#roi-fact').textContent()).includes('没有红点'))throw Error('Non-conflict example mislabeled');
 if(errors.length)throw Error(errors.join(';'));
 console.log(JSON.stringify({status:'PASS',representatives:optionCount,initial,selected,canvas_pixels:rendered,mobile_overflow:false}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
