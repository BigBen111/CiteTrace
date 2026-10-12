// One real question verifies the deployed demo. Credentials never leave this process's memory.
const {chromium}=require('playwright');
const {execFileSync}=require('child_process');
const fs=require('fs');
const path=require('path');
const assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const out=path.join(root,'artifacts','demo-ui');
(async()=>{
 if(!process.env.CITETRACE_TEST_URL)throw Error('Set CITETRACE_TEST_URL to your own deployment; this test may call paid models.');
 if(!process.argv.includes('--allow-paid')&&!process.argv.includes('--resume'))throw Error('Pass --allow-paid to authorize one real model-backed question.');
 fs.mkdirSync(out,{recursive:true});
 const token=execFileSync(path.join(root,'.venv/bin/python'),['-c','from dotenv import dotenv_values; c=dotenv_values("config/.env"); print(c.get("CITETRACE_USER_TOKEN") or c.get("CITETRACE_ADMIN_TOKEN") or "",end="")'],{cwd:root,encoding:'utf8'});
 if(!token)throw Error('No configured access token');
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.CITETRACE_TEST_URL,{waitUntil:'domcontentloaded'});
 assert.equal(await page.locator('#presentButton,.brand-sub,button[onclick="resumeLastJob()"]').count(),0);
 await page.click('#sessionButton');await page.fill('#adminToken',token);await page.click('#connectButton');
 await page.waitForFunction(()=>!document.querySelector('#settingsDialog').open);
 await page.waitForFunction(()=>document.querySelector('#paperReady').textContent.includes('已就绪'));
 await page.screenshot({path:path.join(out,'live-ready.png'),fullPage:true});
 if(process.argv.includes('--resume')){
   const jobs=fs.readdirSync(path.join(root,'artifacts/jobs')).filter(x=>x.endsWith('.json')).map(x=>JSON.parse(fs.readFileSync(path.join(root,'artifacts/jobs',x),'utf8'))).filter(x=>x.question==='论文训练模型时使用了什么 GPU？'&&x.status==='completed').sort((a,b)=>b.created_at.localeCompare(a.created_at));
   if(!jobs.length)throw Error('No previous live check to resume');
   await page.evaluate(id=>localStorage.setItem('citetraceJobId',id),jobs[0].job_id);
   // Read an existing job directly; the recovery button is no longer part of the UI.
   await page.evaluate(()=>resumeLastJob());
 }else{await page.fill('#question','论文训练模型时使用了什么 GPU？');await page.click('#askButton');}
 await page.waitForFunction(()=>!document.querySelector('#askButton').disabled,{},{timeout:180000});
 const summary=await page.locator('#resultSummary').innerText(),answer=await page.locator('#answerText').innerText();
 assert(!summary.includes('未完成'));assert(answer.includes('P100'));
 await page.screenshot({path:path.join(out,'live-answer.png'),fullPage:true});
 await page.setViewportSize({width:1440,height:900});
 await page.screenshot({path:path.join(out,'live-flow-timing.png'),fullPage:true});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert(await page.locator('[data-demo-timing]').count());
 assert(!/演示用时|随机展示|非实测/.test(await page.locator('.progress-panel').innerText()));
 await page.locator('#answerText .citation-link').first().click();
 assert(await page.locator('.evidence-item.highlight').count());
 await page.screenshot({path:path.join(out,'live-evidence.png'),fullPage:true});
 await page.click('#tab-verify');assert((await page.locator('#verifyCount').innerText())!=='0');
 assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'live-check.json'),JSON.stringify({passed:true,url:process.env.CITETRACE_TEST_URL,summary,answer,errors},null,2));
 await browser.close();console.log(JSON.stringify({passed:true,summary,answer}));
})().catch(e=>{console.error(e.message);process.exit(1)});
