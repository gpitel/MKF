# Integrating the FieldSolverBackend into MKF

All edits are against **`OpenMagnetics/MKF` `origin/main`** (function
`MagneticField::calculate_magnetic_field_strength_field`,
`src/physical_models/MagneticField.cpp`, starts at line 247). Line numbers below
are from `git show origin/main:...`. The change is confined to the `.cpp` and the
build files — **`MagneticField.h` is not touched**, so there is no circular include
and no ABI change.

Copy `FieldSolverBackend.h`, `CpuFieldSolver.h`, `CpuFieldSolver.cpp` into
`src/physical_models/`.

---

## Edit 1 — includes (top of MagneticField.cpp, after line 11)

After the existing `#include "support/Utils.h"` block, add:

```cpp
#include "physical_models/FieldSolverBackend.h"
#include "physical_models/CpuFieldSolver.h"
#include "physical_models/GpuFieldSolver.h"
```

## Edit 2 — pick a backend once (inside the function, right after `auto turns = magnetic.get_coil().get_turns_description().value();`, ~line 320)

```cpp
    auto turns = magnetic.get_coil().get_turns_description().value();

    // Field-solve backend for the all-pairs sum below. Try the GPU (WebGPU, WASM
    // build only); it returns nullptr on native / when no adapter is present, and
    // its solve() returns nullopt for non-round wire or unsupported models — either
    // way we fall back to the CPU backend (which OpenMP parallelises).
    CpuFieldSolver cpuFieldSolver;
    auto gpuFieldSolver = GpuFieldSolver::create();  // nullptr on native / no GPU
    FieldSolverBackend* fieldSolverBackend =
        gpuFieldSolver ? static_cast<FieldSolverBackend*>(gpuFieldSolver.get()) : &cpuFieldSolver;
```

## Edit 3 — replace the standard per-turn-pair block

Replace the whole block from

```cpp
        for (auto& inducedFieldPoint : inducedFields[harmonicIndex].get_data()) {
            double totalInducedFieldX = 0;
            double totalInducedFieldY = 0;
            ...
            fieldPoints.push_back(complexFieldPoint);
        }
        complexFieldPerHarmonic[harmonicIndex].set_data(fieldPoints);
```

(origin/main **lines 495–591**, i.e. the code after the ALBACH `continue;`/closing
braces at line 493 down to and including the `set_data(fieldPoints);` at 591)

with:

```cpp
        // --- hoist winding-index lookup out of the O(N^2) loop --------------------
        // Depends only on the inducing point's turn, so compute it once per harmonic
        // instead of once per (induced, inducing) pair. Also required for threading:
        // it is the only get_mutable_coil() call in the hot region.
        const auto& inducingData = inducingFields[harmonicIndex].get_data();
        const auto& inducedData = inducedFields[harmonicIndex].get_data();
        std::vector<std::optional<size_t>> windingIndexPerInducingPoint(inducingData.size(), std::nullopt);
        for (size_t inducingPointIndex = 0; inducingPointIndex < inducingData.size(); ++inducingPointIndex) {
            if (inducingData[inducingPointIndex].get_turn_index()) {
                windingIndexPerInducingPoint[inducingPointIndex] =
                    magnetic.get_mutable_coil().get_winding_index_by_name(
                        turns[inducingData[inducingPointIndex].get_turn_index().value()].get_winding());
            }
        }

        // --- ROSHEN / SULLIVAN gap-fringing seed per induced point ----------------
        // Same math as before; produces the value the sum starts from. Kept serial
        // and here because the magnetizing-current setup mutates operatingPoint.
        std::vector<std::pair<double, double>> inducedSeed;
        if (!isAlbach && (_magneticFieldStrengthFringingEffectModel == MagneticFieldStrengthFringingEffectModels::ROSHEN ||
                          _magneticFieldStrengthFringingEffectModel == MagneticFieldStrengthFringingEffectModels::SULLIVAN)) {
            if (includeFringing && inducedFields[harmonicIndex].get_frequency() == operatingPoint.get_excitations_per_winding()[0].get_frequency()) {
                if (!operatingPoint.get_excitations_per_winding()[0].get_magnetizing_current()) {
                    auto magnetizingInductance = MagneticSimulator().calculate_magnetizing_inductance(operatingPoint, magnetic);
                    auto includeDcCurrent = Inputs::include_dc_offset_into_magnetizing_current(operatingPoint, magnetic.get_turns_ratios());
                    auto magnetizingCurrent = Inputs::calculate_magnetizing_current(operatingPoint.get_mutable_excitations_per_winding()[0],
                                                                                           resolve_dimensional_values(magnetizingInductance.get_magnetizing_inductance()),
                                                                                           true, includeDcCurrent);
                    operatingPoint.get_mutable_excitations_per_winding()[0].set_magnetizing_current(magnetizingCurrent);
                }
                if (!operatingPoint.get_excitations_per_winding()[0].get_magnetizing_current()->get_processed()) {
                    auto excitations = operatingPoint.get_excitations_per_winding();
                    auto magnetizingCurrent = excitations[0].get_magnetizing_current().value();
                    auto processed = Inputs::calculate_basic_processed_data(magnetizingCurrent.get_waveform().value());
                    magnetizingCurrent.set_processed(processed);
                    excitations[0].set_magnetizing_current(magnetizingCurrent);
                    operatingPoint.set_excitations_per_winding(excitations);
                }
                double frequency = inducingFields[harmonicIndex].get_frequency();
                double magneticFieldStrengthGap = get_magnetic_field_strength_gap(operatingPoint, magnetic, frequency);

                inducedSeed.assign(inducedData.size(), {0.0, 0.0});
                for (size_t inducedPointIndex = 0; inducedPointIndex < inducedData.size(); ++inducedPointIndex) {
                    for (auto& gap : gapping) {
                        if (gap.get_coordinates().value()[0] < 0) {
                            continue;
                        }
                        auto complexFieldPoint = _fringingEffectModel->get_magnetic_field_strength_between_gap_and_point(gap, magneticFieldStrengthGap, inducedData[inducedPointIndex]);
                        inducedSeed[inducedPointIndex].first += complexFieldPoint.get_real();
                        inducedSeed[inducedPointIndex].second += complexFieldPoint.get_imaginary();
                        if (std::isnan(complexFieldPoint.get_real())) {
                            throw NaNResultException("NaN found in fringing field calculation");
                        }
                        if (std::isnan(complexFieldPoint.get_imaginary())) {
                            throw NaNResultException("NaN found in fringing field calculation");
                        }
                    }
                }
            }
        }

        // --- dispatch the all-pairs Biot-Savart sum to the backend ----------------
        FieldSolveHarmonicInputs solveInputs{
            inducingData, inducedData, windingIndexPerInducingPoint, inducedSeed,
            coreColumnWidth, coreWidth, coreShapeFamily};
        auto solved = fieldSolverBackend->solve(solveInputs, *_model);
        if (!solved) {                       // backend declined → CPU fallback
            solved = cpuFieldSolver.solve(solveInputs, *_model);
        }
        complexFieldPerHarmonic[harmonicIndex].set_data(solved.value());
```

Behaviour is identical: the seed initialises each induced point's running sum, then
the backend adds the inducing contributions in the original vector order, so the
serial result is bit-for-bit unchanged. The only reordering is that all fringing
seeds are computed before the pair sums (rather than interleaved) — a fringing NaN
still throws first, with the same message.

---

## Edit 4 — CMake (`src/CMakeLists.txt` or wherever MKF sources are listed)

Add the new source and OpenMP:

```cmake
target_sources(MKF PRIVATE
    src/physical_models/CpuFieldSolver.cpp
    src/physical_models/GpuFieldSolver.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MKF PUBLIC OpenMP::OpenMP_CXX)
endif()
```

Native builds then thread automatically; without OpenMP found, the `#pragma` is
ignored and the backend is a plain serial loop (still correct). `GpuFieldSolver`
compiles on native but `create()` returns nullptr there, so native always uses CPU.

### WASM (WebLibMKF / Emscripten) — two independent wins

**(a) OpenMP threading — ship first, helps every design (incl. FOIL like 1164-05-211).**
Emscripten maps `-fopenmp` onto pthreads. Add `-fopenmp -pthread` and a pool, e.g.
`-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency`, and serve cross-origin-isolated
(COOP/COEP) so `SharedArrayBuffer` is available. This alone gives the browser solve
the ~5.5× CPU-backend speedup and is independent of the GPU path.

**(b) GPU backend — round/litz-wire designs only.** `GpuFieldSolver::solve()` awaits
an async WebGPU call, so the Emscripten build needs `-sASYNCIFY` (scope it with
`-sASYNCIFY_ONLY=[...calculate_magnetic_field_strength_field...]` to avoid whole-program
overhead). It calls `Module.webgpuFieldSolve(inp)` — install that in the worker from
the existing `webgpuFieldSolver.js`:

```js
// mkfWorker.js — after libMKF (Module) is ready
import { WebGpuFieldSolver } from './webgpuFieldSolver.js';
let _gpuSolver = null;                       // created lazily, once
Module.webgpuFieldSolve = async (inp) => {   // inp: SoA object built by GpuFieldSolver.cpp
  if (_gpuSolver === null) _gpuSolver = await WebGpuFieldSolver.create();  // null if no adapter
  if (!_gpuSolver) return null;              // → C++ falls back to CPU
  return _gpuSolver.solve(inp);              // { real, imag } Float32Array(Nd)
};
```

If `navigator.gpu` is absent the property can be left unset — `GpuFieldSolver::create()`
checks for it and returns nullptr, keeping the CPU path. (b) is optional and does not
block (a).
