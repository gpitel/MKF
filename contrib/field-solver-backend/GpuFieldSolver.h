#pragma once
// GpuFieldSolver — WebGPU backend for the field solve, for the WASM/browser build.
//
// Implements the same FieldSolverBackend interface as CpuFieldSolver, but offloads
// the all-pairs sum to the GPU via the JS module webgpuFieldSolver.js (validated on
// real hardware: ~124x vs serial, matches f64 to ~1e-8 — see
// process/om-render-tests/test_webgpu_field.js). It packs the borrowed MKF vectors
// into the SoA that module expects and awaits the async WebGPU result.
//
// Scope (v1): round/litz wire only, models LAMMERANER and BINNS_LAWRENSON. That is
// the exact set the JS kernel covers, and it is also the set where the MKF per-pair
// models stay analytic — for a non-round wire (rectangular / FOIL / planar) both
// models delegate to the BINNS *rectangular* path per pair, which the GPU kernel
// does not implement. solve() therefore returns std::nullopt when any winding is
// non-round or the model is unsupported, and MagneticField falls back to the CPU
// backend. (Note: FOIL-wound designs such as 1164-05-211 always take the CPU path;
// the OpenMP CpuFieldSolver is what accelerates them.)
//
// GPU dispatch only exists in the Emscripten build. On a native build create()
// returns nullptr and solve() returns std::nullopt, so native always uses the CPU
// backend (a native GPU path via Dawn/wgpu-native could be added later behind the
// same interface).

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "FieldSolverBackend.h"

namespace OpenMagnetics {

class GpuFieldSolver : public FieldSolverBackend {
public:
    // Probe for a usable GPU. Returns a solver, or nullptr when none is available
    // (native build, no adapter, or the JS bridge is absent) → caller uses CPU.
    // Never throws.
    static std::shared_ptr<GpuFieldSolver> create();

    // Round/litz + LAMMERANER|BINNS → GPU; anything else → std::nullopt (CPU fallback).
    std::optional<std::vector<ComplexFieldPoint>> solve(
        const FieldSolveHarmonicInputs& in, MagneticFieldStrengthModel& model) override;

    std::string name() const override { return "GPU(WebGPU)"; }
};

}  // namespace OpenMagnetics
