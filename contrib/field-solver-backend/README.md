# FieldSolverBackend — pluggable backend for the MKF magnetic-field solve

The C++ refactor that turns the single-threaded O(N²) field solve in
`MagneticField::calculate_magnetic_field_strength_field` into a pluggable backend,
so the same all-pairs Biot-Savart sum can run serial, multithreaded (OpenMP), or on
the GPU — **same results either way**. This is the foundation both proven speedups
land on:

| backend | speedup (1163-turn 1164-05-211) | accuracy | status |
|---|---|---|---|
| CPU serial | 1× (baseline) | exact | this dir, verified |
| CPU + OpenMP | ~5.5× (20 threads) | **bit-identical** | proven in `../fieldbench.cpp` |
| GPU (WebGPU) | ~124× | f32, ~1e-5 rel | proven in `../webgpuFieldSolver.js` + `../test_webgpu_field.js` |

## Why it's slow today

Per harmonic, the solve sums the contribution of every inducing mesh point (each
turn + its 8 mirror images) at every induced point — O(harmonics · turns² · 9),
single-threaded, with a string-based winding-name lookup *inside* the inner loop.
For 1163 turns that's the dominant cost of the simulation.

## The design

`solve()` computes the field at every induced point for one harmonic. The seam
keeps native MKF types on the CPU path (no flattening), so `CpuFieldSolver` is a
mechanical extraction of the existing loop and is provably unchanged. Two structural
improvements fall out for free and are required for correctness under threading:

- the winding-index lookup is **hoisted** out of the O(N²) loop (it depends only on
  the inducing point) — a speedup even before threading;
- parallelism is over **induced points only**, so each point's inner accumulation
  keeps its original order → bit-identical to serial.

A backend returns `std::nullopt` when it can't handle the inputs (e.g. the future
GPU backend on a rectangular/foil or ALBACH model), and the caller falls back to
`CpuFieldSolver`.

## Files

| file | what |
|---|---|
| `FieldSolverBackend.h` | interface + `FieldSolveHarmonicInputs` (borrowed MKF vectors, no copies) |
| `CpuFieldSolver.h/.cpp` | reference backend; serial or OpenMP (`#pragma` ignored if no OpenMP) |
| `GpuFieldSolver.h/.cpp` | WebGPU backend (WASM build); round/litz + LAMMERANER/BINNS, else `nullopt`→CPU. Dispatches to `../webgpuFieldSolver.js` |
| `integration.md` | exact edits to drop it into `MagneticField.cpp` + CMake/WASM/Asyncify flags + the `mkfWorker.js` JS bridge |
| `MagneticField.integration.patch` | the applied edit as a real `git diff` vs `origin/main` (62+/81−), compile-verified with GPU auto-detect wired in |
| `test/` | standalone unit test against mock MKF types (below) |

## Verification status — read this

**1. Logic (`bash test/run.sh`, against mock MKF types in `test/mock/`):**
- `CpuFieldSolver` compiles clean as C++23 (`-Wall -Wextra`, no warnings), serial and OpenMP;
- serial output == an independently-written reference, **bit-exact**;
- OpenMP (8 threads) output == serial, **bit-exact** — the parallel driver is deterministic and thread-safe;
- skip-self, in-core rejection, fringing seed, and turn_index/label propagation all correct.

**2. Real-header compile (against `origin/main` MKF + regenerated `MAS.hpp`, `-Wall -Wextra`):**
- `origin/main` `MagneticField.cpp` baseline type-checks;
- **`MagneticField.cpp` with `MagneticField.integration.patch` applied type-checks** (GPU auto-detect wired in) — the edits are valid against the real MKF/MAS types;
- **`CpuFieldSolver.cpp` type-checks against the real MKF/MAS types** (not just mocks);
- **`GpuFieldSolver.cpp` (native path — packing + wire/model gating) type-checks**, no warnings;
- **`CpuFieldSolver.o` builds with `-fopenmp`.**

The GPU *dispatch* path (the `#ifdef __EMSCRIPTEN__` `emscripten::val` + `.await()`
block) compiles only under `emcc`, and its end-to-end correctness is the browser's
job — but the WGSL kernel it drives is already validated on real hardware
(`../test_webgpu_field.js`). **The GPU backend only accelerates round/litz-wire
designs; FOIL-wound designs like 1164-05-211 return `nullopt` and ride the OpenMP CPU
path — which is the win that matters for them.**

So the logic and the full integration compile are verified. Reproducing step 2
needs a clean tree because the checkout in `workspace/openmagnetics` is on
`feature/insulation-system-requirement`, mid-MAS-migration, and does not self-compile
(its generated `build/MAS/MAS.hpp` is stale against headers already using
`ProcessedWaveform` / `MagneticApplication` / `SubApplication`-as-enum — `MagneticField.o`
fails there *before any edit*). The recipe used here:

```
git worktree add --detach /tmp/mkf-main origin/main          # HEAD 1dccf3fd
git -C /tmp/mkf-main submodule update --init MAS PEAS         # MAS c05a4e6, PEAS ac3c34f
# regenerate MAS.hpp with quicktype (the CMake add_custom_command), then:
g++ -std=c++23 -fsyntax-only -I<worktree>/src -I<masgen> ... MagneticField.cpp CpuFieldSolver.cpp
```

Your dev-container CI does the `MAS.hpp` regen + build automatically, so applying the
patch there is the normal path — this just confirms it compiles before you do.

## How to apply

1. Copy `FieldSolverBackend.h`, `CpuFieldSolver.h`, `CpuFieldSolver.cpp` into
   `MKF/src/physical_models/`.
2. Apply the four edits in `integration.md` (2 includes, 2 locals, replace the
   per-turn-pair block, CMake `find_package(OpenMP)` + add the source).
3. Native build threads automatically. For the browser (WebLibMKF/Emscripten) add
   `-fopenmp -pthread` + a pthread pool and serve cross-origin-isolated — see
   `integration.md`.

## Next (downstream of this seam)

Both backends are written and compile-verified. What remains is build/runtime, which
needs the WebLibMKF Emscripten toolchain (and, for the GPU path, a browser):

- **Land it** — `git apply MagneticField.integration.patch` on an `origin/main`
  branch + the five source files + the CMake edit; let CI build.
- **WASM OpenMP** (`-fopenmp -pthread` + COOP/COEP) — turns on the ~5.5× CPU win in
  the browser for every design, including FOIL. Ship first.
- **WASM GPU** (`-sASYNCIFY` + the `mkfWorker.js` bridge in `integration.md`) — the
  ~124× win for round/litz designs. Optional; does not block the OpenMP win.
