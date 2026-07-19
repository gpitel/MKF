// webgpuFieldSolver.js — GPU backend for the OpenMagnetics magnetic-field solve.
//
// Drop-in module for WebSharedComponents/assets/js. Replaces the O(N^2) CPU loop
// in MagneticField::calculate_magnetic_field_strength_field with a WebGPU compute
// pass. Auto-detects a GPU (navigator.gpu) once; `create()` returns null when no
// usable adapter exists, so the caller keeps the CPU path.
//
// Scope (v1): round-wire analytic models LAMMERANER (0) and BINNS_LAWRENSON (1),
// the round-wire transformer/inductor case. Rectangular/foil (BINNS-rect, WANG)
// and ALBACH (elliptic integrals) stay on CPU — pass model:'cpu'/unsupported and
// the caller falls back.
//
// Inputs mirror the MKF mesh (CoilMesher) as flat SoA. Positions are shared
// across harmonics; only the inducing `value` differs per harmonic. Output is
// Hx/Hy per (harmonic, induced point) in induced-input order, carrying no
// metadata — the caller re-attaches point/turn_index/label by index.

const WGSL = /* wgsl */`
struct Params {
  Ni: u32, Nd: u32, harmonics: u32, model: u32,   // model: 0=Lammeraner, 1=Binns
  twoPi: f32, _a: f32, _b: f32, _c: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
// inducing: packed [x,y,turnLength] * Ni ; [turnIndex,wireIndex] * Ni ; value [h*Ni]
@group(0) @binding(1) var<storage, read> indPos: array<f32>;
@group(0) @binding(2) var<storage, read> indMeta: array<i32>;
@group(0) @binding(3) var<storage, read> indVal: array<f32>;
// induced: packed [x,y] * Nd ; [turnIndex,insideCore] * Nd
@group(0) @binding(4) var<storage, read> ddPos: array<f32>;
@group(0) @binding(5) var<storage, read> ddMeta: array<i32>;
// wire: packed [radius,isRound] * W
@group(0) @binding(6) var<storage, read> wire: array<f32>;
// output: packed [real,imag] * (harmonics*Nd)
@group(0) @binding(7) var<storage, read_write> outv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= P.Nd) { return; }
  let xd = ddPos[d * 2u];
  let yd = ddPos[d * 2u + 1u];
  let td = ddMeta[d * 2u];
  let inCore = ddMeta[d * 2u + 1u];

  for (var h: u32 = 0u; h < P.harmonics; h = h + 1u) {
    var sumX = 0.0;
    var sumY = 0.0;
    for (var i: u32 = 0u; i < P.Ni; i = i + 1u) {
      let ti = indMeta[i * 2u];
      // skip self: induced and inducing are the same turn (MagneticField.cpp:552-555)
      if (ti >= 0 && td >= 0 && ti == td) { continue; }
      // induced point inside the core column (:563) — only when it has no turn
      if (ti >= 0 && td < 0 && inCore != 0) { continue; }

      let ix = indPos[i * 3u];
      let iy = indPos[i * 3u + 1u];
      let dx = xd - ix;
      let dy = yd - iy;
      let dist = sqrt(dx * dx + dy * dy);

      let wi = indMeta[i * 2u + 1u];
      // wire-radius cutoff for round wire (LAMM :939 / BINNS :775)
      if (wi >= 0) {
        let radius = wire[u32(wi) * 2u];
        let isRound = wire[u32(wi) * 2u + 1u];
        if (isRound > 0.5 && dist < radius) { continue; }
      }

      let val = indVal[h * P.Ni + i];
      if (P.model == 1u) {
        // BINNS_LAWRENSON round (:780-783): dx,dy = inducing - induced
        let bdx = ix - xd;
        let bdy = iy - yd;
        let div = P.twoPi * (bdy * bdy + bdx * bdx);
        sumX = sumX + (-val * bdy / div);
        sumY = sumY + ( val * bdx / div);
      } else {
        // LAMMERANER round (:944-948), trig simplified: ex=dx/dist, ey=-dy/dist
        let tl = indPos[i * 3u + 2u];
        let m = -val / (P.twoPi * dist) * tl / sqrt(tl * tl + dist * dist);
        sumX = sumX + m * (dx / dist);
        sumY = sumY + m * (-dy / dist);
      }
    }
    let o = (h * P.Nd + d) * 2u;
    outv[o] = sumX;
    outv[o + 1u] = sumY;
  }
}`;

const TWO_PI = 2 * Math.PI;

/** True if this context exposes WebGPU at all (fast, no device request). */
export function hasWebGpu() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

export class WebGpuFieldSolver {
  constructor(device, adapterInfo) {
    this.device = device;
    this.adapterInfo = adapterInfo || {};
    this.module = device.createShaderModule({ code: WGSL });
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'main' } });
    this._bufs = {};
  }

  /**
   * Probe for a usable GPU. Returns a solver, or null when none is available
   * (→ caller uses the CPU backend). Never throws.
   */
  static async create() {
    try {
      if (!hasWebGpu()) return null;
      let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      if (!device) return null;
      return new WebGpuFieldSolver(device, adapter.info || {});
    } catch (_) {
      return null;
    }
  }

  _buf(key, byteLength, usage) {
    let b = this._bufs[key];
    if (!b || b.size < byteLength) {
      if (b) b.destroy();
      b = this.device.createBuffer({ size: Math.max(byteLength, 16), usage });
      this._bufs[key] = b;
    }
    return b;
  }

  /**
   * @param {object} inp SoA field-solve inputs (see module header). model:
   *   'lammeraner'|'binns'. Returns {real, imag}: Float32Array(harmonics*Nd),
   *   or null if the model is unsupported on GPU (caller falls back to CPU).
   */
  async solve(inp) {
    const modelId = inp.model === 'binns' ? 1 : (inp.model === 'lammeraner' ? 0 : -1);
    if (modelId < 0) return null;         // unsupported model → CPU
    const { Ni, Nd, harmonics } = inp;
    const dev = this.device;

    // Pack SoA (kept off the hot path; the WASM would hand these over directly).
    const indPos = new Float32Array(Ni * 3);
    const indMeta = new Int32Array(Ni * 2);
    for (let i = 0; i < Ni; i++) {
      indPos[i * 3] = inp.indX[i]; indPos[i * 3 + 1] = inp.indY[i]; indPos[i * 3 + 2] = inp.indTurnLength ? inp.indTurnLength[i] : 1;
      indMeta[i * 2] = inp.indTurnIndex ? inp.indTurnIndex[i] : -1;
      indMeta[i * 2 + 1] = inp.indWireIndex ? inp.indWireIndex[i] : -1;
    }
    const ddPos = new Float32Array(Nd * 2);
    const ddMeta = new Int32Array(Nd * 2);
    for (let d = 0; d < Nd; d++) {
      ddPos[d * 2] = inp.ddX[d]; ddPos[d * 2 + 1] = inp.ddY[d];
      ddMeta[d * 2] = inp.ddTurnIndex ? inp.ddTurnIndex[d] : -1;
      ddMeta[d * 2 + 1] = inp.ddInsideCore ? inp.ddInsideCore[d] : 0;
    }
    const W = inp.wireRadius ? inp.wireRadius.length : 0;
    const wire = new Float32Array(Math.max(W, 1) * 2);
    for (let w = 0; w < W; w++) { wire[w * 2] = inp.wireRadius[w]; wire[w * 2 + 1] = inp.wireIsRound[w] ? 1 : 0; }
    const outLen = harmonics * Nd * 2;

    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
    const bIndPos = this._buf('indPos', indPos.byteLength, S | D);
    const bIndMeta = this._buf('indMeta', indMeta.byteLength, S | D);
    const bIndVal = this._buf('indVal', inp.indValue.byteLength, S | D);
    const bDdPos = this._buf('ddPos', ddPos.byteLength, S | D);
    const bDdMeta = this._buf('ddMeta', ddMeta.byteLength, S | D);
    const bWire = this._buf('wire', wire.byteLength, S | D);
    const bOut = this._buf('out', outLen * 4, S | C);
    const bRead = this._buf('read', outLen * 4, GPUBufferUsage.MAP_READ | D);

    dev.queue.writeBuffer(bIndPos, 0, indPos);
    dev.queue.writeBuffer(bIndMeta, 0, indMeta);
    dev.queue.writeBuffer(bIndVal, 0, inp.indValue);
    dev.queue.writeBuffer(bDdPos, 0, ddPos);
    dev.queue.writeBuffer(bDdMeta, 0, ddMeta);
    dev.queue.writeBuffer(bWire, 0, wire);

    const uni = new ArrayBuffer(32);
    new Uint32Array(uni, 0, 4).set([Ni, Nd, harmonics, modelId]);
    new Float32Array(uni, 16, 4).set([TWO_PI, 0, 0, 0]);
    const bUni = this._buf('uni', 32, GPUBufferUsage.UNIFORM | D);
    dev.queue.writeBuffer(bUni, 0, new Uint8Array(uni));

    const bind = dev.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bUni } },
        { binding: 1, resource: { buffer: bIndPos } },
        { binding: 2, resource: { buffer: bIndMeta } },
        { binding: 3, resource: { buffer: bIndVal } },
        { binding: 4, resource: { buffer: bDdPos } },
        { binding: 5, resource: { buffer: bDdMeta } },
        { binding: 6, resource: { buffer: bWire } },
        { binding: 7, resource: { buffer: bOut } },
      ],
    });

    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(Nd / 64));
    pass.end();
    enc.copyBufferToBuffer(bOut, 0, bRead, 0, outLen * 4);
    dev.queue.submit([enc.finish()]);

    await bRead.mapAsync(GPUMapMode.READ);
    const packed = new Float32Array(bRead.getMappedRange().slice(0));
    bRead.unmap();

    const real = new Float32Array(harmonics * Nd);
    const imag = new Float32Array(harmonics * Nd);
    for (let k = 0; k < harmonics * Nd; k++) { real[k] = packed[k * 2]; imag[k] = packed[k * 2 + 1]; }
    return { real, imag };
  }

  dispose() {
    for (const k in this._bufs) { try { this._bufs[k].destroy(); } catch (_) {} }
    this._bufs = {};
  }
}
