'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
require('../docs/data.js');
const {waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('real Chrome202queries3030targetcontrols, band identity, signed weights, keyboard, invalidation, exports and mobile', {timeout:120000}, async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'kernellab-browser-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{
  for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),`Chrome debugger did not start: ${startupError?.message||diagnostics||browser.exitCode}`);const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.enable');await call('Page.navigate',{url:process.env.KERNEL_RESPONSE_URL||'file://'+path.resolve(__dirname,'../docs/index.html')});
  for(let i=0;i<250;i++){if(await js(`Boolean(globalThis.KERNEL_APP?.getWitness())`))break;await delay(20);}
  assert.equal(await js(`document.getElementById('error').textContent`),'');
  let count=0;
  for(const arm of ['length_0_8','length_5']){
    const refs=[0,1,2,3].flatMap(k=>JSON.parse(fs.readFileSync(path.join(__dirname,'../kernellab/artifacts/response-'+arm+'-refs-'+k+'.json'))));
    const edited=JSON.parse(fs.readFileSync(path.join(__dirname,'../kernellab/artifacts/response-'+arm+'.json'))).unit_target_edit_curves;
    for(let i=0;i<101;i++){
      await js(`document.getElementById('arm').value='${arm}';document.getElementById('arm').dispatchEvent(new Event('change'));document.getElementById('query').value='${KERNEL_DATA.query[i][0]}';document.getElementById('query').dispatchEvent(new Event('input'));`);
      const r=await js(`KERNEL_APP.getWitness().response`);assert.ok(Math.abs(r.original_mean-refs[i].original_mean)<2e-10);r.weights.forEach((v,j)=>assert.ok(Math.abs(v-refs[i].weights[j])<2e-10));
      // One renderer call reads15 independent target edits at this query, then real UI checks each target.
      for(let j=0;j<15;j++){
        const r=await js(`(()=>{document.getElementById('target').value='${j}';document.getElementById('target').dispatchEvent(new Event('change'));const w=KERNEL_APP.getWitness();return {mean:w.response.edited_mean,variance:w.response.latent_variance,display:Number(document.getElementById('edited-mean').textContent),terms:document.querySelectorAll('#terms tr').length};})()`);
        assert.ok(Math.abs(r.mean-edited[j].means[i])<2e-10);assert.ok(Math.abs(r.variance-refs[i].latent_variance)<2e-10);assert.ok(Math.abs(r.display-r.mean)<1e-6*(1+Math.abs(r.mean)));assert.equal(r.terms,15);count++;
      }
    }
  }assert.equal(count,3030);
  await js(`document.getElementById('query-slider').value='0';document.getElementById('query-slider').dispatchEvent(new Event('input'));document.getElementById('query-slider').focus()`);await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});assert.equal(await js(`KERNEL_APP.getWitness().row[0]`),.01);
  const before=await js(`KERNEL_APP.getWitness().response`);await js(`document.getElementById('band').value='latent';document.getElementById('band').dispatchEvent(new Event('change'))`);assert.equal(await js(`KERNEL_APP.getWitness().response.edited_mean`),before.edited_mean);assert.match(await js(`document.getElementById('band-caption').textContent`),/Excludes observation noise/);
  await js(`document.getElementById('query').value='';document.getElementById('query').dispatchEvent(new Event('input'))`);assert.equal(await js(`KERNEL_APP.getWitness()`),null);assert.equal(await js(`document.getElementById('edited-mean').textContent`),'-');await js(`document.getElementById('download').click()`);assert.match(await js(`document.getElementById('copy-status').textContent`),/Verify an input/);await js(`document.getElementById('reset').click()`);assert.equal(await js(`KERNEL_APP.getWitness().response.mean_shift`),0);
  for(const value of ['', '-2.01', '2.01']){await js(`document.getElementById('delta').value='${value}';document.getElementById('delta').dispatchEvent(new Event('input'))`);assert.equal(await js(`KERNEL_APP.getWitness()`),null);assert.equal(await js(`document.getElementById('edited-mean').textContent`),'-');}await js(`document.getElementById('reset').click()`);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async()=>{throw Error('denied')}},configurable:true});document.getElementById('copy').click()`);await delay(10);assert.match(await js(`document.getElementById('copy-status').textContent`),/Copy was blocked/);
  await js(`Object.defineProperty(navigator,'clipboard',{value:{writeText:async s=>globalThis.copied=s},configurable:true});document.getElementById('copy').click()`);await delay(10);assert.equal(await js(`JSON.parse(globalThis.copied).response.mean_shift`),0);
  await js(`HTMLAnchorElement.prototype.click=function(){globalThis.downloaded={name:this.download,url:this.href}};document.getElementById('download').click()`);assert.equal(await js(`globalThis.downloaded.name`),'kernellab-target-response.json');assert.equal(await js(`fetch(globalThis.downloaded.url).then(r=>r.json()).then(p=>p.model.rows.length)`),15);
  for(const width of [320,390,1280]){await js(`for(const e of document.querySelectorAll('details'))e.open=true`);await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});await delay(20);assert.ok(await js(`document.documentElement.scrollWidth<=innerWidth`),'overflow at '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,select,summary,input')).every(e=>e.getBoundingClientRect().height>=44)`),'touch at '+width);}
  await js(`for(const e of document.querySelectorAll('details'))e.open=false;document.getElementById('arm').value='length_0_8';document.getElementById('arm').dispatchEvent(new Event('change'));document.getElementById('delta').value='1';document.getElementById('delta').dispatchEvent(new Event('input'));document.getElementById('band').value='observation';document.getElementById('band').dispatchEvent(new Event('change'));document.activeElement.blur();window.scrollTo(0,0)`);
  if(process.env.KERNEL_RESPONSE_EXPAND)await js(`for(const e of document.querySelectorAll('details'))e.open=true`);
  if(process.env.KERNEL_RESPONSE_SCREENSHOT){const width=Number(process.env.KERNEL_RESPONSE_SCREENSHOT_WIDTH||1280);await call('Emulation.setDeviceMetricsOverride',{width,height:1200,deviceScaleFactor:1,mobile:width<700});await delay(30);const metrics=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:metrics.cssContentSize.height,scale:1}});fs.writeFileSync(process.env.KERNEL_RESPONSE_SCREENSHOT,Buffer.from(image.data,'base64'));}
 }finally{if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
