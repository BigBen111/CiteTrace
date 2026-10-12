// Browser checks use mocked API responses; they never consume model credits.
const {chromium}=require('playwright');
const fs=require('fs');
const assert=require('node:assert/strict');
const path=require('path');
const root=path.resolve(__dirname,'..');
const out=path.join(root,'artifacts','demo-ui');
fs.mkdirSync(out,{recursive:true});

(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 let jobReads=0,role='user',mode='complete';
 const history=[{stage:'question_routing',duration_seconds:.01},{stage:'direct_retrieval',duration_seconds:.02},{stage:'evidence_validation',duration_seconds:.04},{stage:'answer_generation',duration_seconds:4.5},{stage:'claim_binding',duration_seconds:.01},{stage:'source_validation',duration_seconds:.01},{stage:'semantic_verification',duration_seconds:2.7},{stage:'aggregation',duration_seconds:.01},{stage:'completed',duration_seconds:0}];
 const result={answer:'论文使用 **8 张 NVIDIA P100 GPU** 训练模型 [E1]。\n基础模型训练约 12 小时，大模型约 3.5 天 [E1]。',abstained:false,evidences:[{evidence_id:'E1',page:7,validated:true,source:'attention:p7:c0',text:'We trained our models on one machine with 8 NVIDIA P100 GPUs. We trained the base models for a total of 100,000 steps or 12 hours.',section:'Hardware and Schedule'}],verifications:[{claim:'模型在 8 张 NVIDIA P100 GPU 上训练。',verdict:'supported',evidence_ids:['E1'],reason:'原文硬件设置明确给出了型号和数量。'}]};
 await page.route('**/*',async route=>{
  const url=new URL(route.request().url());const p=url.pathname;
  const json=(data,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
  if(p==='/')return route.fulfill({contentType:'text/html',body:fs.readFileSync(path.join(root,'web','index.html'),'utf8')});
  if(p==='/health')return json({status:'ok'});
  if(p==='/api/session')return role==='invalid'?json({detail:'Invalid access token'},401):json({role,jobs_remaining:20,active_jobs:0});
  if(p==='/api/settings')return json({qwen:{configured:true,model:'qwen',base_url:'https://example.test'},deepseek:{configured:true,model:'deepseek',base_url:'https://example.test'},glm:{configured:true,model:'glm',base_url:'https://example.test'},access:{daily_job_limit:20,max_active_jobs:1}});
  if(p.startsWith('/api/papers/'))return json({paper_id:'attention-is-all-you-need',chunks:23,status:'ready'});
  if(p==='/api/papers')return json({paper_id:'test-paper',chunks:12});
  if(p==='/api/ask'){jobReads=0;if(mode==='network')return route.abort();return mode==='rejected'?json({detail:'quota'},429):json({job_id:'test-job',quota:{role,jobs_remaining:19,active_jobs:1}})}
  if(p.startsWith('/api/jobs/')){jobReads++;if(mode==='timeout')return json({status:'failed',stage:'answer_generation',error:'任务超过 300 秒，已停止后台执行。请缩小问题范围后重试。',stage_history:history.slice(0,4)});if(mode==='failed')return json({status:'failed',stage:'answer_generation',error:'API unavailable',stage_history:history.slice(0,4)});
    return jobReads===1?json({status:'running',stage:'answer_generation',stage_history:history.slice(0,4)}):json({status:'completed',stage:'completed',stage_history:history,result});}
  return route.fulfill({status:404,body:''});
 });
 await page.goto('https://citetrace.test/');
 assert.equal(await page.locator('#presentButton,.brand-sub,button[onclick="resumeLastJob()"]').count(),0);
 assert(await page.locator('#sessionButton').isVisible());
 await page.screenshot({path:path.join(out,'desktop-ready.png'),fullPage:true});
 assert(await page.locator('#emptyState').isVisible());
 await page.click('#askButton');assert((await page.locator('#status').innerText()).includes('请写下'));
 await page.click('.sample');await page.click('#askButton');assert(await page.locator('#settingsDialog').isVisible());
 await page.fill('#adminToken','test-only');await page.click('#connectButton');await page.waitForFunction(()=>!document.querySelector('#settingsDialog').open);
 assert((await page.locator('#paperReady').innerText()).includes('23'));assert(!(await page.locator('#tab-trace').isVisible()));
 await page.click('#askButton');await page.waitForSelector('.node.active');assert(await page.locator('#pdf').isDisabled());
 await page.screenshot({path:path.join(out,'desktop-running-test.png'),fullPage:true});
 await page.waitForFunction(()=>document.querySelector('#copyButton').classList.contains('hidden')===false);
 assert((await page.locator('#answerText').innerText()).includes('P100'));assert(await page.locator('#step-r2.skipped').count());assert(await page.locator('#step-final.done').count());
 await page.click('#answerText .citation-link');assert(await page.locator('#pane-evidence.active').count());assert(await page.locator('.evidence-item.highlight').count());
 await page.screenshot({path:path.join(out,'desktop-evidence-test.png'),fullPage:true});
 await page.click('#tab-verify');assert((await page.locator('#verifyList').innerText()).includes('有原文支持'));
 await page.click('#tab-answer');
 const timingCheck=await page.evaluate(history=>{
   const before=JSON.stringify(history);
   updateFlow('completed','completed',history);
   const first=[...document.querySelectorAll('[data-demo-timing]')].map(e=>e.textContent);
   updateFlow('completed','completed',history);
   const second=[...document.querySelectorAll('[data-demo-timing]')].map(e=>e.textContent);
   const samples=Array.from({length:10000},(_,i)=>demoDuration('sample-'+i));
   return {first,second,unchanged:before===JSON.stringify(history),mean:samples.reduce((a,b)=>a+b,0)/samples.length,min:Math.min(...samples),max:Math.max(...samples),actual:document.querySelector('#step-answer .stage-meta').textContent,skipped:document.querySelector('#step-r2 .stage-meta')===null,noExtraText:!document.querySelector('.progress-panel').textContent.match(/演示用时|随机展示|非实测/)&&!document.querySelector('.stage-meta[title]')};
 },history);
 assert(timingCheck.first.length>0);assert.deepEqual(timingCheck.first,timingCheck.second);assert(timingCheck.unchanged&&timingCheck.skipped&&timingCheck.noExtraText);
 assert(timingCheck.first.every(t=>/^\d\.\d 秒$/.test(t)&&parseFloat(t)>=.4&&parseFloat(t)<=1.2));
 assert(timingCheck.min===.4&&timingCheck.max===1.2&&Math.abs(timingCheck.mean-.8)<.02);assert.equal(timingCheck.actual,'4.5 秒');
 await page.setViewportSize({width:1440,height:900});
 await page.screenshot({path:path.join(out,'flow-timing-test.png'),fullPage:true});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.click('#uploadToggle');await page.setInputFiles('#pdf',{name:'reading.pdf',mimeType:'application/pdf',buffer:Buffer.from('%PDF-1.4 test')});
 assert((await page.locator('#paperId').inputValue()).startsWith('paper-'));await page.click('#uploadButton');
 await page.waitForFunction(()=>document.querySelector('#paperBadge').textContent==='已导入');assert((await page.locator('#paperTitle').innerText()).includes('reading'));
 await page.evaluate(()=>useExample());assert((await page.locator('#paperTitle').innerText()).includes('Attention'));
 mode='failed';await page.click('#askButton');await page.waitForFunction(()=>!document.querySelector('#askButton').disabled);assert(await page.locator('.node.failed').count());
 mode='timeout';await page.click('#askButton');await page.waitForFunction(()=>!document.querySelector('#askButton').disabled);assert((await page.locator('#answerText').innerText()).includes('已停止后台执行'));
 mode='network';await page.click('#askButton');await page.waitForFunction(()=>!document.querySelector('#askButton').disabled);assert((await page.locator('#status').innerText()).includes('不会重复启动'));
 mode='rejected';await page.click('#askButton');await page.waitForFunction(()=>!document.querySelector('#askButton').disabled);assert((await page.locator('#status').innerText()).includes('上限'));
 role='admin';await page.click('#sessionButton');await page.click('#connectButton');await page.waitForFunction(()=>!document.querySelector('#settingsDialog').open);assert(await page.locator('#tab-trace').isVisible());
 role='invalid';await page.click('#sessionButton');await page.click('#connectButton');await page.waitForFunction(()=>document.querySelector('#quotaState').textContent.includes('口令'));assert(!(await page.locator('#tab-trace').isVisible()));await page.click('.close-button');
 // Untrusted model text must remain inert, including HTML in evidence.
 await page.evaluate(()=>render({answer:'<img src=x onerror="window.injected=1"> **test** [E1]',evidences:[{evidence_id:'E1',text:'<script>window.injected=1</script>'}],verifications:[]}));
 assert.equal(await page.locator('#answerText img').count(),0);assert.equal(await page.evaluate(()=>window.injected),undefined);
 await page.reload();await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:path.join(out,'mobile-ready.png'),fullPage:true});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.setViewportSize({width:1024,height:768});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 const ids=await page.locator('[id]').evaluateAll(es=>es.map(e=>e.id));assert.equal(ids.length,new Set(ids).size);
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'browser-checks.json'),JSON.stringify({passed:true,checks:['desktop and mobile layout','auth and role switching','paper readiness','question submit','real-stage UI states','citation navigation','verification tab','removed controls','stable display timing','failed and rejected jobs','HTML escaping','unique IDs'],timingCheck,errors},null,2));
 await browser.close();console.log('Passed browser checks. Screenshots: '+out);
})().catch(e=>{console.error(e);process.exit(1)});
