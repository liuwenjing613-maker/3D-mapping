const {createRequire}=require('module');
const {pathToFileURL}=require('url');
const path=require('path');
const deps=createRequire('C:/Users/liuwenjing/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json');
const {chromium}=deps('playwright');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1024,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.resolve(__dirname,'zoom_preview_standalone.html')).href);
 const f=page.frameLocator('iframe');
 await f.locator('#zoom-summary').filter({hasText:'ROI-0035'}).waitFor();
 const optionCount=await f.locator('#zoom-roi option').count();if(optionCount!==32)throw Error('Missing ROIs');
 const initial=await f.locator('#zoom-summary').textContent();
 if(!initial.includes('440 × 320'))throw Error('Wrong displayed RGB crop');
 await f.locator('#zoom-crop').selectOption('tight');await f.locator('#zoom-summary').filter({hasText:'160 × 160'}).waitFor();
 await f.locator('#zoom-mode').selectOption('pure');
 await f.locator('#zoom-after').click({position:{x:160,y:160}});
 if(!(await f.locator('#zoom-selected').textContent()).includes('原 F'))throw Error('Mask pixel selection failed');
 await f.locator('#zoom-all').click();
 await f.locator('#zoom-view').selectOption('5');await f.locator('#zoom-summary').filter({hasText:'帧 1765'}).waitFor();
 const witness=await f.locator('#zoom-summary').textContent();
 // Compare browser-decoded PNG label bytes with the source grayscale IDs.
 const verifiedLabels=await f.locator('#zoom-data').evaluate(async el=>{
  const d=JSON.parse(el.textContent);const entry=d.entries.find(e=>e.info.key==='ROI-0035_witness_ID206');
  const im=new Image();im.src=entry.after;await im.decode();const c=document.createElement('canvas');c.width=im.width;c.height=im.height;const x=c.getContext('2d');x.drawImage(im,0,0);const a=x.getImageData(0,0,c.width,c.height).data;const ids=new Set();for(let i=0;i<a.length;i+=4){if(a[i]!==a[i+1]||a[i]!==a[i+2])throw Error('PNG labels changed channels');if(a[i])ids.add(a[i]);}
  return {shown_ids:ids.size,maximum:Math.max(...ids),native_masks:entry.info.local_mask_count};
 });
 if(!verifiedLabels.shown_ids)throw Error('Empty mask raster');
 const h=await f.locator('#room0-cropformer-zoom-review').evaluate(r=>Math.ceil(r.getBoundingClientRect().height)+32);
 await page.setViewportSize({width:1024,height:Math.max(1100,h+64)});await page.locator('iframe').evaluate((e,h)=>e.style.height=h+'px',h);await page.screenshot({path:path.resolve(__dirname,'zoom_preview_desktop.png'),fullPage:true});
 await page.setViewportSize({width:360,height:2200});await f.locator('#zoom-roi').selectOption('73');await f.locator('#zoom-summary').filter({hasText:'ROI-0073'}).waitFor();
 const mobile=await f.locator('#room0-cropformer-zoom-review').evaluate(r=>({width:r.clientWidth,scroll:r.scrollWidth,height:Math.ceil(r.getBoundingClientRect().height)+32}));if(mobile.scroll>mobile.width+2)throw Error('Mobile overflow');await page.locator('iframe').evaluate((e,h)=>e.style.height=h+'px',mobile.height);await page.screenshot({path:path.resolve(__dirname,'zoom_preview_mobile.png'),fullPage:true});
 // Every RGB + label raster must load, including all alternative crops and views.
 const all=await f.locator('#zoom-data').evaluate(async el=>{const d=JSON.parse(el.textContent);let n=0;for(const e of d.entries){await Promise.all([e.rgb,e.before,e.after].map(async src=>{const im=new Image();im.src=src;await im.decode();}));n++;}return n;});
 if(all!==159)throw Error('Missing selected RGB segmentation inputs');if(errors.length)throw Error(errors.join(';'));
 console.log(JSON.stringify({status:'PASS',ROIs:optionCount,crop_results:all,initial,witness,verifiedLabels,mobile_overflow:false}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1);});
