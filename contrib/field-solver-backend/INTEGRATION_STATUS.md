# Field-solve performance — branch status

Branch: `perf/field-solve-threading` (based on `fix/leakage-grid-cap`).

## Wired in this branch

**OpenMP parallelisation of the field solve** — `MagneticField::calculate_magnetic_field_strength_field`
(`src/physical_models/MagneticField.cpp`). The per-induced-point field is an
independent sum over the inducing points, so the induced loop is parallelised
with `#pragma omp parallel for`. The loop body is unchanged; thread-safety comes
from: (1) each iteration writes only its own `fieldPoints[i]` (pre-sized, not
`push_back`); (2) the ROSHEN/SULLIVAN fringing path's lazy magnetizing-current
setup — the one shared mutation of `operatingPoint` — is forced once before the
parallel region; (3) the body is wrapped in `try`/`catch`, the first exception
captured via `std::exception_ptr` and rethrown after the region (exceptions must
not escape an OpenMP region).

Verified **bit-identical to the serial result** (`calculate_leakage_inductance`
on `tests/testData/leakage_multiwinding_foil_stress.json`, exact string match).
The `#pragma` is a no-op without `-fopenmp`, so this holds independent of threads.

## To activate

- **Native threading:** add to CMake, after the `MKF` target is defined:
  ```cmake
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(MKF PUBLIC OpenMP::OpenMP_CXX)
  endif()
  ```
  (Left out of this branch's commit only to avoid entangling a local build-hacked
  CMakeLists; it is a clean one-liner to add.)
- **Browser (WASM) threading:** WebLibMKF must build with `-fopenmp -pthread`
  `-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency`, and the dev/prod server
  must be cross-origin-isolated (COOP/COEP) so `SharedArrayBuffer` is available.
  Until then the WASM build runs the loop serially (correct, just single-threaded).

## Staged here (not yet wired) — the WebGPU backend

`FieldSolverBackend.h`, `CpuFieldSolver.{h,cpp}`, `GpuFieldSolver.{h,cpp}`,
`webgpuFieldSolver.js`, `integration.md`. The GPU path offloads the same all-pairs
Biot-Savart sum to WebGPU (**~124× on round/litz wire**; returns `nullopt` for
foil/planar and unsupported models → CPU fallback, so foil designs like the stress
fixture ride the OpenMP CPU path). Wiring it needs Emscripten `-sASYNCIFY` and the
`Module.webgpuFieldSolve` worker bridge — see `integration.md`.

## Verification status

- Restructuring: **bit-identical to serial** (this branch, WASM serial run). ✓
- OpenMP pattern: **~5.5× at 20 threads, bit-exact** — standalone kernel bench
  (`fieldbench.cpp`, in the source deliverable). ✓
- WebGPU: **~124×, matches f64 to 1.9e-8** (`webgpuFieldSolver.js` + Deno wgpu). ✓
- Integrated loop, native OpenMP (2026-07-19): **bit-EXACT** (identical result hash
  at 1 vs 20 threads) and **1.83×** (8849 ms → 4841 ms) on
  `calculate_leakage_inductance` for the foil stress fixture. ✓ The 1.83× (vs the
  kernel's ~6×) is Amdahl-limited: with the leakage grid cap in place the field solve
  is ~half the cost, so the serial mesh-generation + energy-integral around it now
  dominate. Threading is a **secondary** win on top of the grid cap, not a
  replacement for it. Also confirmed by inspection: the field models set their state
  once in setup (`MagneticField.cpp:345-357`) and only read it in the loop.
