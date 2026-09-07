// Validates webgpuFieldSolver.js against a CPU reference (f64) implementing the
// identical kernel + branches, on synthetic multi-winding data that exercises
// every branch: skip-self, induced-in-core, wire-radius cutoff, fringing points
// (wireIndex<0), both models. Proves the GPU kernel is correct.
//
// Run: deno run --unstable-webgpu -A test_webgpu_field.js
import { WebGpuFieldSolver, hasWebGpu } from './webgpuFieldSolver.js';

const TWO_PI = 2 * Math.PI;

// ---- build synthetic multi-winding inputs -----------------------------------
function makeInputs() {
  const W = 3;                                  // windings
  const wireRadius = new Float32Array([0.00035, 0.00025, 0.0005]);
  const wireIsRound = new Uint32Array([1, 1, 1]);
  const turnsPerWinding = [120, 120, 60];
  const harmonics = 4;

  // lay turns out on a grid, assign winding, current
  const turns = [];
  let col = 0;
  for (let w = 0; w < W; w++) {
    for (let t = 0; t < turnsPerWinding[w]; t++) {
      const layer = (col % 24);
      const x = 0.014 + layer * 0.0007;
      const y = -0.04 + ((col * 37) % 100) * 0.0008;
      turns.push({ x, y, winding: w, cur: [4.0, 1.3, 0.5][w] });
      col++;
    }
  }
  const n = turns.length;

  // induced = 1 per turn (turn centers) + a few core/leakage points (td<0)
  const ddX = [], ddY = [], ddTI = [], ddCore = [];
  for (let t = 0; t < n; t++) { ddX.push(turns[t].x); ddY.push(turns[t].y); ddTI.push(t); ddCore.push(0); }
  // leakage-style points with no turn index; some flagged inside-core
  for (let k = 0; k < 10; k++) { ddX.push(0.0 + k * 0.001); ddY.push(0.0); ddTI.push(-1); ddCore.push(k % 2); }
  const Nd = ddX.length;

  // inducing = 9 mirror images per turn (+ a couple fringing points, wireIndex<0)
  const minx = Math.min(...turns.map(t => t.x)), maxx = Math.max(...turns.map(t => t.x));
  const miny = Math.min(...turns.map(t => t.y)), maxy = Math.max(...turns.map(t => t.y));
  const Wd = (maxx - minx) * 1.2 + 1e-3, Hd = (maxy - miny) * 1.2 + 1e-3;
  const indX = [], indY = [], indTL = [], indTI = [], indWI = [];
  const baseVal = [];        // unscaled current per inducing point
  for (let t = 0; t < n; t++) {
    for (let kx = -1; kx <= 1; kx++) for (let ky = -1; ky <= 1; ky++) {
      const sx = kx === 0 ? 1 : -1, sy = ky === 0 ? 1 : -1;
      indX.push(sx * turns[t].x + kx * 2 * Wd);
      indY.push(sy * turns[t].y + ky * 2 * Hd);
      indTL.push(0.09 + turns[t].winding * 0.01);
      indTI.push(t);
      indWI.push(turns[t].winding);
      baseVal.push(sx * sy * turns[t].cur);
    }
  }
  // fringing points: no turn, no wire
  for (let k = 0; k < 4; k++) { indX.push(0.001 * k); indY.push(0.03); indTL.push(1); indTI.push(-1); indWI.push(-1); baseVal.push(0.2); }
  const Ni = indX.length;

  // per-harmonic inducing values (harmonic falloff)
  const indValue = new Float32Array(harmonics * Ni);
  for (let h = 0; h < harmonics; h++) for (let i = 0; i < Ni; i++) indValue[h * Ni + i] = baseVal[i] / (h + 1);

  return {
    Ni, Nd, harmonics, W,
    indX: Float32Array.from(indX), indY: Float32Array.from(indY),
    indTurnLength: Float32Array.from(indTL), indTurnIndex: Int32Array.from(indTI), indWireIndex: Int32Array.from(indWI),
    indValue,
    ddX: Float32Array.from(ddX), ddY: Float32Array.from(ddY),
    ddTurnIndex: Int32Array.from(ddTI), ddInsideCore: Uint32Array.from(ddCore),
    wireRadius, wireIsRound,
  };
}

// ---- CPU reference: identical kernel + branches, f64 ------------------------
function cpuSolve(inp, model) {
  const { Ni, Nd, harmonics } = inp;
  const real = new Float64Array(harmonics * Nd), imag = new Float64Array(harmonics * Nd);
  for (let d = 0; d < Nd; d++) {
    const xd = inp.ddX[d], yd = inp.ddY[d], td = inp.ddTurnIndex[d], inCore = inp.ddInsideCore[d];
    for (let h = 0; h < harmonics; h++) {
      let sx = 0, sy = 0;
      for (let i = 0; i < Ni; i++) {
        const ti = inp.indTurnIndex[i];
        if (ti >= 0 && td >= 0 && ti === td) continue;
        if (ti >= 0 && td < 0 && inCore !== 0) continue;
        const ix = inp.indX[i], iy = inp.indY[i];
        const dx = xd - ix, dy = yd - iy, dist = Math.sqrt(dx * dx + dy * dy);
        const wi = inp.indWireIndex[i];
        if (wi >= 0 && inp.wireIsRound[wi] && dist < inp.wireRadius[wi]) continue;
        const val = inp.indValue[h * Ni + i];
        if (model === 'binns') {
          const bdx = ix - xd, bdy = iy - yd, div = TWO_PI * (bdy * bdy + bdx * bdx);
          sx += -val * bdy / div; sy += val * bdx / div;
        } else {
          const tl = inp.indTurnLength[i];
          const m = -val / (TWO_PI * dist) * tl / Math.sqrt(tl * tl + dist * dist);
          sx += m * (dx / dist); sy += m * (-dy / dist);
        }
      }
      real[h * Nd + d] = sx; imag[h * Nd + d] = sy;
    }
  }
  return { real, imag };
}

function compare(label, gpu, cpu) {
  let maxAbs = 0, sumRef = 0, sumErr = 0, maxRel = 0;
  for (let k = 0; k < cpu.real.length; k++) {
    for (const [g, c] of [[gpu.real[k], cpu.real[k]], [gpu.imag[k], cpu.imag[k]]]) {
      const e = Math.abs(g - c); maxAbs = Math.max(maxAbs, e);
      sumErr += e; sumRef += Math.abs(c);
      if (Math.abs(c) > 1e-9) maxRel = Math.max(maxRel, e / Math.abs(c));
    }
  }
  const aggRel = sumErr / sumRef;
  const pass = aggRel < 1e-4 && maxRel < 5e-3;   // f32 tolerance
  console.log(`${label.padEnd(12)} points=${cpu.real.length}  maxAbs=${maxAbs.toExponential(2)}  maxRel=${maxRel.toExponential(2)}  aggRel=${aggRel.toExponential(2)}  -> ${pass ? 'PASS' : 'FAIL'}`);
  return pass;
}

// ---- run --------------------------------------------------------------------
if (!hasWebGpu()) { console.log('navigator.gpu unavailable -> (in the app this falls back to CPU)'); Deno.exit(0); }
const solver = await WebGpuFieldSolver.create();
if (!solver) { console.log('no GPU adapter -> (falls back to CPU)'); Deno.exit(0); }
console.log('GPU: ' + JSON.stringify(solver.adapterInfo));

const inp = makeInputs();
console.log(`inputs: windings=${inp.W} inducing=${inp.Ni} induced=${inp.Nd} harmonics=${inp.harmonics}`);
let ok = true;
for (const model of ['lammeraner', 'binns']) {
  inp.model = model;
  const gpu = await solver.solve(inp);
  const cpu = cpuSolve(inp, model);
  ok = compare(model, gpu, cpu) && ok;
}
// unsupported model -> null (CPU fallback)
inp.model = 'albach';
const un = await solver.solve(inp);
console.log('unsupported model returns null: ' + (un === null ? 'PASS' : 'FAIL'));
ok = ok && un === null;

solver.dispose();
console.log(ok ? '\nALL PASS' : '\nFAILURES');
Deno.exit(ok ? 0 : 1);
