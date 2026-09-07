// Deno WebGPU port of the MKF field solve (LAMMERANER round-wire), same data +
// math as the CPU fieldbench, to validate correctness + measure GPU time.
// Run: deno run --unstable-webgpu -A fieldbench_webgpu.js turns.txt
const CPU_CHECKSUM = 43389468.1753538;   // fieldbench f64 serial
const CPU_SERIAL_MS = 2044.0;
const HARMONICS = 4, TURN_LENGTH = 0.1;

const WGSL = `
struct Params { Ni: u32, Nd: u32, harmonics: u32, _p: u32, turnLength: f32, _a: f32, _b: f32, _c: f32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> ind_x: array<f32>;
@group(0) @binding(2) var<storage, read> ind_y: array<f32>;
@group(0) @binding(3) var<storage, read> ind_v: array<f32>;
@group(0) @binding(4) var<storage, read> dd_x: array<f32>;
@group(0) @binding(5) var<storage, read> dd_y: array<f32>;
@group(0) @binding(6) var<storage, read_write> outv: array<f32>;
const PI: f32 = 3.14159265358979;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= P.Nd) { return; }
  let xd = dd_x[d]; let yd = dd_y[d];
  for (var h: u32 = 0u; h < P.harmonics; h = h + 1u) {
    let hscale = 1.0 / f32(h + 1u);
    var sumX = 0.0; var sumY = 0.0;
    for (var i: u32 = 0u; i < P.Ni; i = i + 1u) {
      if ((i / 9u) == d && (i % 9u) == 4u) { continue; }
      let dx = xd - ind_x[i];
      let dy = yd - ind_y[i];
      let dist = sqrt(dx*dx + dy*dy);
      let val = ind_v[i] * hscale;
      let m = -val / (2.0 * PI * dist) * P.turnLength / sqrt(P.turnLength*P.turnLength + dist*dist);
      sumX = sumX + m * (dx / dist);
      sumY = sumY + m * (-dy / dist);
    }
    let o = (h * P.Nd + d) * 2u;
    outv[o] = sumX; outv[o + 1u] = sumY;
  }
}`;

const path = Deno.args[0] || "turns.txt";
const lines = Deno.readTextFileSync(path).trim().split(/\r?\n/);
const n = parseInt(lines[0]);
const tx = new Float64Array(n), ty = new Float64Array(n), tv = new Float64Array(n);
let minx=1e30,maxx=-1e30,miny=1e30,maxy=-1e30;
for (let i=0;i<n;i++){ const p=lines[i+1].split(' '); tx[i]=+p[0]; ty[i]=+p[1]; tv[i]=+p[2];
  minx=Math.min(minx,tx[i]);maxx=Math.max(maxx,tx[i]);miny=Math.min(miny,ty[i]);maxy=Math.max(maxy,ty[i]); }
const W=(maxx-minx)*1.2+1e-3, H=(maxy-miny)*1.2+1e-3;
const Ni=n*9, Nd=n;
const ind_x=new Float32Array(Ni), ind_y=new Float32Array(Ni), ind_v=new Float32Array(Ni);
let k=0;
for (let i=0;i<n;i++) for (let kx=-1;kx<=1;kx++) for (let ky=-1;ky<=1;ky++){
  const sx=kx===0?1:-1, sy=ky===0?1:-1;
  ind_x[k]=sx*tx[i]+kx*2*W; ind_y[k]=sy*ty[i]+ky*2*H; ind_v[k]=sx*sy*tv[i]; k++;
}
const dd_x=new Float32Array(n), dd_y=new Float32Array(n);
for (let i=0;i<n;i++){ dd_x[i]=tx[i]; dd_y[i]=ty[i]; }

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) { console.log("no GPU adapter -> would fall back to CPU"); Deno.exit(0); }
const info = adapter.info || {};
const device = await adapter.requestDevice();
console.log("GPU adapter: " + (info.vendor||"?") + " / " + (info.device||info.architecture||info.description||"?") + (adapter.isFallbackAdapter ? " [FALLBACK/software]" : ""));

const S=GPUBufferUsage.STORAGE, C=GPUBufferUsage.COPY_SRC, D=GPUBufferUsage.COPY_DST;
const mk=(arr,u)=>{ const b=device.createBuffer({size:arr.byteLength,usage:u}); device.queue.writeBuffer(b,0,arr); return b; };
const bIndX=mk(ind_x,S|D),bIndY=mk(ind_y,S|D),bIndV=mk(ind_v,S|D),bDdX=mk(dd_x,S|D),bDdY=mk(dd_y,S|D);
const outLen=Nd*HARMONICS*2;
const bOut=device.createBuffer({size:outLen*4,usage:S|C});
const bRead=device.createBuffer({size:outLen*4,usage:GPUBufferUsage.MAP_READ|D});
const uni=new ArrayBuffer(32); new Uint32Array(uni,0,4).set([Ni,Nd,HARMONICS,0]); new Float32Array(uni,16,4).set([TURN_LENGTH,0,0,0]);
const bUni=mk(new Uint8Array(uni),GPUBufferUsage.UNIFORM|D);
const mod=device.createShaderModule({code:WGSL});
const pipe=device.createComputePipeline({layout:'auto',compute:{module:mod,entryPoint:'main'}});
const bind=device.createBindGroup({layout:pipe.getBindGroupLayout(0),entries:[
  {binding:0,resource:{buffer:bUni}},{binding:1,resource:{buffer:bIndX}},{binding:2,resource:{buffer:bIndY}},
  {binding:3,resource:{buffer:bIndV}},{binding:4,resource:{buffer:bDdX}},{binding:5,resource:{buffer:bDdY}},{binding:6,resource:{buffer:bOut}}]});

async function dispatch(){
  const enc=device.createCommandEncoder(); const pass=enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0,bind); pass.dispatchWorkgroups(Math.ceil(Nd/64)); pass.end();
  device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
}
await dispatch(); // warm-up
const t0=performance.now(); await dispatch(); const ms=performance.now()-t0;

const enc=device.createCommandEncoder(); enc.copyBufferToBuffer(bOut,0,bRead,0,outLen*4); device.queue.submit([enc.finish()]);
await bRead.mapAsync(GPUMapMode.READ);
const res=new Float32Array(bRead.getMappedRange().slice(0)); bRead.unmap();
let checksum=0; for (let i=0;i<res.length;i++) checksum+=Math.abs(res[i]);
const relErr=Math.abs(checksum-CPU_CHECKSUM)/CPU_CHECKSUM;
console.log("turns="+n+"  inducing="+Ni+"  induced="+Nd+"  harmonics="+HARMONICS+"  pair_evals="+(Nd*Ni*HARMONICS/1e6).toFixed(2)+"M");
console.log("GPU  time_ms="+ms.toFixed(2)+"  checksum="+checksum.toFixed(6));
console.log("CPU  time_ms="+CPU_SERIAL_MS.toFixed(1)+" (serial)  checksum="+CPU_CHECKSUM.toFixed(6));
console.log("speedup="+(CPU_SERIAL_MS/ms).toFixed(1)+"x   rel_err(f32 vs f64)="+relErr.toExponential(3)+"   -> "+(relErr<2e-4?"MATCH (within f32 tolerance)":"MISMATCH"));
