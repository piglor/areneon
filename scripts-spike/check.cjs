// SPIKE: DOM-stub check. node check.js page.html
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8');
const src=html.match(/<script>([\s\S]*)<\/script>/)[1];
function stub(){const o={_s:{},style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},children:[],value:'',checked:true,innerHTML:'',textContent:'',
 addEventListener(){},appendChild(){},querySelector(){return stub();},remove(){},scrollIntoView(){},offsetWidth:1,scrollHeight:1,open:true,parentNode:null};return o;}
const els={};
const document={getElementById:id=>els[id]||(els[id]=stub()),createElement:()=>stub(),body:{classList:{toggle(){}}}};
const ctx={document,console,setTimeout:()=>0,clearTimeout(){}};ctx.globalThis=ctx;
vm.createContext(ctx);vm.runInContext(src,ctx);
const R=ctx.__replay;const D=R.DATA;
let tot={k:0,f:0};const kinds={};
D.demos.forEach((d,di)=>d.views.forEach(v=>{
  const b=R.buildSteps(v.lines,v.viewer);tot.k+=b.known;tot.f+=b.flagged;Object.assign(kinds,b.flaggedKinds);
  b.steps.forEach(s=>{const vm_=R.viewModel(s.state,v.viewer,D);assert(vm_.top.slots.length===2&&vm_.bottom.slots.length===2);
    for(const sd of [vm_.top,vm_.bottom])for(const c of sd.slots)if(!c.empty){assert(c.hpPct>=0&&c.hpPct<=100,'pct');assert(['#30a46c','#d9a400','#e5484d'].includes(c.hpColor));if(c.fainted)assert(c.hpPct===0);}
    R.decisionHTML(s.state);});
  console.log(d.id,v.viewer,'steps',b.steps.length,'known',b.known,'flagged',b.flagged,'hidden',b.hidden);
}));
// pure assertions on golden
const g=D.demos[0].views[0];const b=R.buildSteps(g.lines,'p1');
const gi=b.steps.findIndex(s=>/Foe Charizard used Flamethrower on Venusaur/.test(s.text));assert(gi>0,'move text');
const dm=b.steps[gi+2];assert(/lost 152 HP - 7\/159/.test(dm.text),dm.text);
const vmx=R.viewModel(dm.state,'p1',D);assert.strictEqual(vmx.bottom.slots[0].hpColor,'#e5484d');assert.strictEqual(vmx.bottom.slots[0].types.join(),'grass,poison');
assert.strictEqual(vmx.top.slots[0].exact,false);
const last=b.steps[b.steps.length-1].state;assert(last.winner||last.tie);
assert.strictEqual(R.hpColor(50),'#d9a400');assert.strictEqual(R.hpColor(51),'#30a46c');assert.strictEqual(R.hpColor(20),'#e5484d');
assert.strictEqual(JSON.stringify(R.parseCond('50/100g')),'{"hp":50,"max":100,"status":null}');assert.strictEqual(R.parseCond('0 fnt').status,'fnt');
// unknown line is flagged, not dropped
const u=R.buildSteps(['|gametype|doubles','|weirdkind|x'],'p1');assert(u.flagged===1&&u.steps[1].flagged&&u.steps[1].text==='|weirdkind|x');
// UI path
D.demos.forEach((d,di)=>{['both','p1','p2'].forEach(m=>{R.App.setDemo(di,m);const n=R.App.built[0].steps.length;for(let i=0;i<n;i++)R.App.goto(i);R.App.goto(0);R.App.play();R.App.stop();});});
console.log('TOTAL known',tot.k,'flagged',tot.f,'flagged kinds',JSON.stringify(kinds));
console.log('OK');
