# GPU compute backend for the OpenMagnetics field solve — feasibility & design

**Goal:** keep the CPU path as default, add an optional GPU path for the O(N²)
magnetic-field solve, selectable at deploy time ("keep both, toggle it on").

**One-line answer:** the toggle mechanism is easy (a `Settings` enum already
plumbed to JS and Python); the GPU *kernel* is a real port but the math is
GPU-friendly; the catch is **the heavy compute runs in the user's browser, not
on a server — so a *docker* toggle only helps a server-side path that doesn't
exist today.** The lever that actually speeds up the app is **WebGPU compute in
the browser**, toggled by a frontend setting, not docker.

---

## 1. What we'd accelerate

`MagneticField::calculate_magnetic_field_strength_field`
(`MKF/src/physical_models/MagneticField.cpp:495-590`) — the all-pairs
Biot–Savart sum: for each harmonic, for each induced point, sum the field from
every inducing point (every turn + its 9 mirror images). ~O(harmonics ×
turns² × 9). It dominates winding-loss and leakage-inductance time and is the
reason a 1163-turn design takes minutes.

**It is an ideal GPU workload** (N-body). One thread per induced point; each
reads the shared inducing array and writes its own output — no cross-thread
reduction. For round wire the two main models collapse to trig-free closed
forms:

```
BINNS_LAWRENSON (round):  dx,dy = inducing - induced;  D = 2π(dx²+dy²)
                          Hx = -I·dy/D    Hy = I·dx/D
LAMMERANER (round):       adds a finite-turn-length factor tl/√(tl²+dist²)
```

Branches are few and predicated (skip-self, in-core, wire-radius cutoff) — GPU
friendly. **Caveat:** the *default* model is **ALBACH**, which uses iterative
complete elliptic integrals (AGM, up to 100 iters/filament) — a much harder,
branch-divergent kernel. A first GPU cut should route round-wire work through
LAMMERANER/BINNS (equivalent for concentric round windings) and leave ALBACH on
CPU.

Data crossing the boundary is small and flat (SoA): per harmonic, inducing
`{x,y,value,turnLength,turnIndex,wireIndex}` (~9N), induced `{x,y,turnIndex,
insideCore}` (~N), uniforms `wireRadius[W]`; output `{Hx,Hy}` per induced point.
Full spec in the appendix.

## 2. The architectural reality (this changes the plan)

| Path | Where it runs | Can docker toggle it? |
|---|---|---|
| **`simulate` + core adviser** (the heavy EM compute) | **Browser WASM worker** (`taskQueue.js:1490` → `mkfWorker.js` → `libMKF.wasm`) | **No** — it's on the end-user's machine |
| Server plots / materials / PDF | FastAPI + Celery worker (PyMKF `plot_*`, FreeCAD, LaTeX) | Yes, but it's not the EM sim |
| **Native PyMKF** (your `magneticdesigner`) | In-process `.pyd` | N/A (no docker), env var works |

The dev stack (`docker-compose.yml`) has **no GPU passthrough** anywhere (no
`runtime: nvidia`, no `--gpus`). More importantly, even if it did, it couldn't
touch `simulate` — that executes in the browser.

**Consequence:** there are three deployment targets and the GPU story differs
for each. "Docker toggle" is only meaningful for a *new* server-side compute
service.

## 3. Auto-detect, no toggle (preferred)

Skip the docker/manual switch entirely: **probe for a usable GPU once at init;
use it if present, else CPU.** No config, no env var, no user setting to get
wrong — "keep both" falls out of a capability probe + guaranteed CPU fallback.

- **Browser (WebGPU):** in `mkfWorker.js`, `const a = navigator.gpu &&
  await navigator.gpu.requestAdapter();` — an adapter (+ `requestDevice()`
  succeeding) means GPU; `navigator.gpu` undefined or a throw means CPU. Probe
  once, cache the backend. (WebGPU needs a secure context — https/localhost —
  and ships in Chrome/Edge 113+, Safari 18+, Firefox 141+.)
- **Native / PyMKF:** at engine init the GPU backend tries to acquire a device
  (Dawn/wgpu-native, or CUDA); on any failure the field solver keeps the CPU
  backend. `Settings::get_compute_backend()` still exists but defaults to
  `auto`; an explicit `cpu`/`gpu` override is worth keeping **only for
  debugging / CI determinism**, not as the primary UX.

The Settings enum is still the seam (already plumbed to JS `set_settings` at
`taskQueue.js:697/1694` and Python `PyMKF.set_settings`), but in `auto` the
value is *computed by probing*, not supplied by a docker flag.

### Two refinements that make auto-detect robust
1. **Fall back on runtime failure, not just a missing adapter.** Route to CPU
   when: no adapter/device; pipeline init fails; the selected model isn't
   GPU-ported (ALBACH → CPU for now); or a one-time self-check disagrees with
   CPU beyond tolerance. A GPU that's *present* isn't proof the path *works*.
2. **Use a size threshold.** GPU dispatch + buffer upload has fixed overhead, so
   small designs are faster on CPU. `auto` should mean *GPU when available AND
   the problem is big enough* (e.g. turns > ~200 — same reasoning as the
   instanced-render threshold); tiny coils stay on CPU. Otherwise a GPU would
   make small designs slightly *slower*.

## 4. GPU backend options

| Backend | Runs in browser? | Runs native? | Vendor | Notes |
|---|---|---|---|---|
| **WebGPU (WGSL)** | ✅ (the whole point) | ✅ via Dawn / wgpu-native | Any (NV/AMD/Intel/Apple) | **One kernel serves browser *and* native.** No server, no CUDA lock-in. |
| **CUDA** | ❌ | ✅ | NVIDIA only | Max raw perf; needs nvidia-docker for a server path |
| **SYCL / OpenCL** | ❌ | ✅ | Cross-vendor native | More portable than CUDA, less mature toolchain here |

**Recommendation: WebGPU is the primary backend.** It's the only option that
runs where the compute actually is (the browser), on the GPU that's *already*
drawing the 3D coil, cross-vendor, offline, no server. And the same WGSL kernel
can later back a **native** Dawn/wgpu-native backend so PyMKF/native gets GPU
too — literally "keep both" (CPU + GPU) with a single kernel implementation.
CUDA is worth adding only as a max-performance NVIDIA server option *if* you go
the server-compute route.

### Precision caveat (matters for "same results")
GPUs are f32-fast / f64-slow. A WebGPU compute kernel in **f32** matches the CPU
(f64) field to ~1e-4 relative — **"same to engineering tolerance," not
bit-identical.** That's fine for loss/field numbers, but it's a real difference
from the CPU-threading path (which *is* bit-identical). If you ever need
bit-exactness, that stays on CPU.

## 5. "Keep both" — the concrete architecture

1. **Seam:** in `calculate_magnetic_field_strength_field`, replace the inner
   double loop with a call to a `FieldSolverBackend` interface. `CpuBackend`
   (today's loop, + OpenMP threading from the prior benchmark) and
   `WebGpuBackend` / `CudaBackend` implement it.
2. **Selection:** `Settings::get_compute_backend()` (`auto|cpu|webgpu|cuda`) with
   runtime capability probe + CPU fallback on init failure or unsupported model
   (e.g. ALBACH stays CPU for now).
3. **Verification:** reuse the `fieldbench` checksum approach — assert GPU output
   matches CPU within tolerance on a fixture set in CI.

## 5b. Prototype result (measured)

A standalone WebGPU compute port of the LAMMERANER round-wire kernel
(`fieldbench_webgpu.js`, WGSL), run on the same 1163-turn geometry as the CPU
`fieldbench` (48.7M pair-evals, 4 harmonics), on this machine's NVIDIA GPU via
Deno/wgpu:

| | Time | Checksum |
|---|---|---|
| CPU serial (`fieldbench`) | 2044 ms | 43389468.175354 |
| **WebGPU (real GPU)** | **16.4 ms** | 43389468.997848 |

**≈125× faster, and the result matches to 1.9×10⁻⁸ relative** (aggregate;
individual field points are f32, ~1e-4). This is the exact dominant loop of the
simulation, proving the browser-GPU path is both correct and dramatically
faster. (Headless Chromium exposed no WebGPU adapter — the auto-detect correctly
fell back to CPU there; the number above came from Deno's native wgpu hitting
the real GPU.)

## 6. Effort & phasing

| Phase | Work | Payoff | Effort |
|---|---|---|---|
| **0** | CPU **OpenMP** on the loop (already benchmarked: 5.5× same-results) | Every non-browser path faster now | **Low** |
| **1** | **WebGPU** backend (WGSL LAMMERANER/BINNS) dispatched from `mkfWorker.js`; frontend toggle + adapter detection | **Browser sim 10–100× on the client GPU** | **High** |
| **2** | Native **Dawn/wgpu-native** backend reusing the WGSL → PyMKF/`magneticdesigner` GPU | Your local PyMKF work faster | Med |
| **3** *(optional)* | **CUDA** backend + **server `simulate` endpoint** + nvidia-docker | The only place a **docker toggle** is real | Very high (also moves compute server-side) |

## 7. Direct answer on the "docker toggle"

A docker env var (`MKF_COMPUTE_BACKEND=cpu|cuda`) is trivial to wire into
`docker-compose.yml` `environment:` blocks and read in PyMKF/Settings — **but it
only steers server-side PyMKF**, which today does plots/materials, not the EM
sim. To make a docker GPU toggle meaningful for the actual simulation you must
first **move `simulate` server-side** (a new WebBackend endpoint calling a
CUDA-built PyMKF) — that's Phase 3, and it trades away the app's current
client-only/offline design (data upload, latency, a GPU server per deployment).

For the browser app as it exists, **"keep both, toggle GPU on" = a WebGPU
backend + a frontend setting**, not docker. The Settings plumbing makes the
toggle itself a one-key change; the kernel is the real work.

---

## Appendix — GPU kernel I/O (round-wire, per harmonic)

Inputs (SoA): `ind_x[Ni], ind_y[Ni], ind_value[Ni], ind_turnLength[Ni],
ind_turnIndex[Ni], ind_wireIndex[Ni]` (Ni≈9·turns); `indd_x[Nd], indd_y[Nd],
indd_turnIndex[Nd], indd_insideCore[Nd]` (Nd≈turns); uniforms `wireRadius[W],
wireIsRound[W]`. Output: `out_real[Nd], out_imag[Nd]`. CPU re-attaches
`point[]`/`turn_index`/`label` (order-preserving) to rebuild `ComplexFieldPoint`.
Output must carry `turn_index` (proximity loss filters by it) and preserve input
order (leakage patches by index).

LAMMERANER kernel (one thread per induced point d):
```
sumX=sumY=0; xd=indd_x[d]; yd=indd_y[d]; td=indd_turnIndex[d]; inCore=indd_insideCore[d]
for i in 0..Ni:
    ti=ind_turnIndex[i]
    if ti>=0 && td>=0 && ti==td: continue                 # skip self
    if ti>=0 && td<0  && inCore:  continue                 # induced inside core
    dx=xd-ind_x[i]; dy=yd-ind_y[i]; dist=sqrt(dx*dx+dy*dy)
    w=ind_wireIndex[i]
    if w>=0 && wireIsRound[w] && dist<wireRadius[w]: continue
    tl=ind_turnLength[i]
    mod=-ind_value[i]/(2*PI*dist)*tl/sqrt(tl*tl+dist*dist)
    sumX += mod*(dx/dist);  sumY += mod*(-dy/dist)
out_real[d]=sumX; out_imag[d]=sumY
```
