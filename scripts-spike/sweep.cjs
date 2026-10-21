const fs=require('fs'),vm=require('vm');
const src=fs.readFileSync(process.argv[2],'utf8').match(/<script>([\s\S]*)<\/script>/)[1];
const ctx={console};ctx.globalThis=ctx;ctx.__NOINIT=true;vm.createContext(ctx);vm.runInContext(src,ctx);
const R=ctx.__replay;const dir='fixtures/showdown/gen9championsvgc2026regmc/random/';let k=0,f=0,kinds={},exc=0;
for(const n of fs.readdirSync(dir).filter(x=>/^b\d+\.json$/.test(x))){const d=JSON.parse(fs.readFileSync(dir+n));
 for(const v of ['p1','p2']){try{const b=R.buildSteps(d.channels[v].lines,v);b.steps.forEach(s=>R.viewModel(s.state,v,R.DATA));k+=b.known;f+=b.flagged;for(const x in b.flaggedKinds)kinds[x]=(kinds[x]||0)+b.flaggedKinds[x];}catch(e){exc++;console.log(n,v,e.message)}}}
console.log('known',k,'flagged',f,JSON.stringify(kinds),'exceptions',exc);
