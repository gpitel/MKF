#pragma once
// CpuFieldSolver — the reference field-solve backend. A mechanical extraction of
// the all-pairs loop from MagneticField::calculate_magnetic_field_strength_field,
// so its serial output is bit-identical to the pre-refactor engine.
//
// When built with OpenMP (-fopenmp / find_package(OpenMP)), the per-induced-point
// sweep is parallelised. Parallelism is over induced points only; each point's
// inner accumulation keeps its original order, so results stay bit-identical to the
// serial path (verified standalone in process/om-render-tests/fieldbench.cpp:
// 20 threads → 5.5x, checksum unchanged). Without OpenMP the pragma is ignored and
// this compiles/behaves as a plain serial loop.

#include <optional>
#include <string>
#include <vector>

#include "FieldSolverBackend.h"

namespace OpenMagnetics {

class CpuFieldSolver : public FieldSolverBackend {
public:
    // Always handles the inputs (never returns nullopt) — it is the fallback.
    std::optional<std::vector<ComplexFieldPoint>> solve(
        const FieldSolveHarmonicInputs& in, MagneticFieldStrengthModel& model) override;

    std::string name() const override { return "CPU"; }
};

}  // namespace OpenMagnetics
