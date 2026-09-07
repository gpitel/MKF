# WebGPU field-solve integration plan (browser path)

Wiring the proven WGSL field kernel (`fieldbench_webgpu.js`, ~125× on real GPU,
results match to 1.9e-8) into the OpenMagnetics browser engine, with GPU
auto-detected and a guaranteed CPU fallback.

Prereq context: the sim runs client-side in the WASM Web Worker
(`taskQueue.js:1490` → `mkfWorker.js` → `libMKF.wasm`); the hot loop is
`MagneticField::calculate_magnetic_field_strength_field`
(`MKF/src/physical_models/MagneticField.cpp:495-590`). Full data seam in
`gpu-backend-feasibility.md`.

---

## The one hard problem: async GPU vs sync WASM

WebGPU is asynchronous (device init, `mapAsync` readback all return Promises).
The WASM `simulate()` runs to completion synchronously inside the worker, and the
field solve is buried deep in its call stack
(`simulate → calculate_winding_losses → WindingLosses::calculate_losses →
calculate_magnetic_field_strength_field`). You cannot call an async GPU op from
synchronous C++ without one of:

- **(A) Emscripten WebGPU + scoped Asyncify** — compile WASM with `-sUSE_WEBGPU`;
  the C++ owns the GPU via `webgpu.h`; `-sASYNCIFY_ONLY=[field call chain]`
  suspends/resumes the WASM stack across the one async readback per solve. Device
  is created once at startup (JS side) and adopted by the module
  (`Module.preinitializedWebGPUDevice`), so only the per-solve readback is async.
  - **+** one C++ codebase; the same `WebGpuBackend` also targets **native** via
    Dawn/wgpu-native → browser and PyMKF share it. Matches the `FieldSolverBackend`
    design. Minimal API-surface change.
  - **−** Asyncify adds stack save/restore overhead to the instrumented call
    chain and some code size; the chain list must be maintained.

- **(B) JS-orchestrated split** — the worker drives the sequence: WASM builds the
  mesh → JS runs the WGSL (the proven harness) → WASM finishes losses given the
  injected field. No Asyncify.
  - **+** reuses the working JS/WGSL kernel as-is; worker is naturally async.
  - **−** invasive C++ refactor: `simulate` / `calculate_losses` / leakage must
    split into *pre-field* and *post-field* halves exposed as separate WASM
    exports, and the field result must be marshalled back in. Browser-only (no
    native reuse). Two WASM round-trips + copies per solve.

**Recommendation: (A)** as the target architecture — one kernel serving browser +
native, cleanest fit with the auto-detect Settings design — but **de-risk it with
a Phase-1 that reuses the (B)-style JS harness for validation only**, so the
kernel is proven in the real browser before touching Asyncify.

---

## Phases

### Phase 0 — done
Proven WGSL kernel + CPU baseline (`fieldbench.cpp`, `fieldbench_webgpu.js`):
2044 ms → 16 ms, match 1.9e-8.

### Phase 1 — in-browser correctness (low risk, no sim restructure)
Goal: prove the kernel + WebGPU worker plumbing in the actual browser, against
the WASM's own field output.
- **C++/WASM:** add one read-only export `get_field_solve_inputs(operatingPoint,
  magnetic)` in `WebLibMKF/src/libMKF.cpp` that runs `CoilMesher` +
  harmonic-current scaling and returns the flat SoA arrays (inducing
  `x,y,value,turnLength,turnIndex,wireIndex`; induced `x,y,turnIndex,insideCore`;
  uniforms) — wraps existing mesh code, no algorithm change.
- **Worker:** in `WebSharedComponents/assets/js/mkfWorker.js`, add WebGPU init
  (`navigator.gpu.requestAdapter()` → device → pipeline from a new
  `assets/js/wgsl/fieldSolve.wgsl`) and a `computeFieldGpu(inputs)` that runs the
  kernel (port of `fieldbench_webgpu.js`).
- **Validate:** for a fixture set, compare `computeFieldGpu` vs the WASM
  `calculate_magnetic_field_strength_field` output within tolerance. Ship nothing
  yet — this is the go/no-go on browser WebGPU availability + kernel fidelity.

### Phase 2 — C++ backend seam + GPU dispatch
- **`MagneticField.{h,cpp}`:** extract the `:495-590` loop behind a
  `FieldSolverBackend` interface; `CpuFieldSolver` = today's loop (+ OpenMP for
  native from the earlier benchmark). `calculate_magnetic_field_strength_field`
  selects the backend, builds SoA once, calls `backend->solve(...)`, rebuilds
  `ComplexFieldPoint`s (order-preserving; carries `turn_index`).
- **`WebGpuFieldSolver`:** implemented with `webgpu.h` (Emscripten) / Dawn
  (native), same WGSL. Build changes in `WebLibMKF/CMakeLists.txt`:
  `-sUSE_WEBGPU=1`, `-sASYNCIFY -sASYNCIFY_ONLY=@asyncify_list.txt`, adopt the
  worker-created device.
- **Scope guards:** round-wire LAMMERANER/BINNS only in v1; **ALBACH (the default
  model) stays CPU** (elliptic integrals) — the backend returns `unsupported` and
  the caller keeps CPU. Leakage inductance uses the same field function → benefits
  automatically.

### Phase 3 — auto-detect + fallback + threshold
- `Settings` gains `compute_backend` (`auto|cpu|gpu`), default `auto`
  (`MKF/src/support/Settings.h`; surfaced via existing `get/set_settings`).
- Worker probes `navigator.gpu` once at init; on adapter+device success sets
  `gpu`, else `cpu`. **Fall back to CPU on:** no adapter, pipeline/init failure,
  unsupported model, or a one-time self-check that disagrees with CPU beyond
  tolerance.
- **Size threshold:** GPU only above ~200 turns (dispatch/upload overhead makes
  small designs faster on CPU) — same rationale as the instanced-render threshold.

### Phase 4 — verify + roll out
- CI: extend the checksum harness — GPU path vs CPU path within tolerance on a
  design matrix (round E/U cores, multi-winding, leakage). Assert identical
  `turn_index`/ordering so proximity-loss and leakage consumers are unaffected.
- Ship behind a default-on `auto` with an easy kill-switch (`compute_backend=cpu`)
  for support.

---

## Files touched (summary)

| Layer | File | Change |
|---|---|---|
| Engine | `MKF/src/physical_models/MagneticField.{h,cpp}` | `FieldSolverBackend` seam; CPU + GPU impls |
| Engine | `MKF/src/support/Settings.h` | `compute_backend` enum + accessors |
| Engine (new) | `MKF/src/physical_models/WebGpuFieldSolver.*`, `fieldSolve.wgsl` | GPU backend + shader |
| WASM | `WebLibMKF/src/libMKF.cpp` | `get_field_solve_inputs` export (Phase 1); device adoption |
| WASM build | `WebLibMKF/CMakeLists.txt` | `-sUSE_WEBGPU`, scoped `-sASYNCIFY` |
| Worker | `WebSharedComponents/assets/js/mkfWorker.js` (+ `mkfRuntime.js`) | WebGPU init, adapter probe, pipeline, `computeFieldGpu` |
| Frontend | — | none required (auto); optional "GPU active" indicator |

## Risks & mitigations
- **Asyncify overhead on `simulate`** → scope with `ASYNCIFY_ONLY` to the field
  chain; benchmark code-size + non-GPU regression; if unacceptable, fall to
  option (B).
- **f32 precision** → results match to ~1e-4 per point (engineering-fine for
  losses), not bit-exact; CPU stays the bit-exact default. Documented, checked in
  CI with tolerance.
- **WebGPU availability** (older browsers, blocked GPUs, secure-context) → the
  auto-detect fallback makes this a non-event; CPU path unchanged.
- **ALBACH default model** → unsupported on GPU v1; either route round-wire
  through LAMMERANER/BINNS (equivalent for concentric round windings) or port the
  elliptic-integral kernel later.

## Effort
Phase 1: ~days (1 export + worker plumbing, reuses proven kernel). Phase 2: the
bulk (backend seam + Emscripten WebGPU + Asyncify). Phases 3–4: small. Native
(Dawn) backend + PyMKF reuse is an add-on once Phase 2's abstraction exists.
