#pragma once
// FieldSolverBackend — pluggable backend for the O(N^2) magnetic-field summation
// in MagneticField::calculate_magnetic_field_strength_field.
//
// The hot path is, per harmonic, an all-pairs Biot-Savart sum: for every induced
// mesh point, accumulate the contribution of every inducing mesh point (turn +
// its 8 mirror images). Cost is O(harmonics * inducedPoints * inducingPoints),
// i.e. ~O(harmonics * turns^2 * 9). For the 1163-turn 1164-05-211 transformer this
// dominates the solve. This interface isolates exactly that sum behind a virtual
// `solve()` so it can be run:
//   * serially           (CpuFieldSolver, default)               — bit-identical
//   * multithreaded       (CpuFieldSolver with OpenMP)            — bit-identical
//   * on the GPU          (GpuFieldSolver, round-wire models)     — f32, ~1e-5 rel
// A backend returns std::nullopt when it cannot handle the inputs (e.g. the GPU
// backend on a rectangular/foil or ALBACH model), and the caller falls back.
//
// The seam keeps native MKF types (FieldPoint / ComplexFieldPoint / the model) —
// no flattening on the CPU path — so the serial backend is a mechanical extraction
// of the existing loop and provably unchanged. The GPU backend does its own SoA
// packing internally (mirrors process/om-render-tests/webgpuFieldSolver.js).

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include "physical_models/MagneticField.h"   // FieldPoint, ComplexFieldPoint,
                                              // MagneticFieldStrengthModel, CoreShapeFamily

namespace OpenMagnetics {

// Below this many induced points the all-pairs solve is faster done serially than
// paying the fixed cost of spawning threads (tens of ms one-time) or a GPU dispatch
// (~12-16 ms) — see process/om-render-tests/bench/RESULTS.md, where the OpenMP and
// GPU wins only appear past ~250 turns. Small solves stay on the serial path.
inline constexpr std::size_t FIELD_SOLVE_PARALLEL_MIN = 200;

// Free function defined in MagneticField.cpp (external linkage). Declared here so
// backends in their own translation unit can reuse the one definition.
bool is_inside_core(FieldPoint inducedFieldPoint, double coreColumnWidth, double coreWidth,
                    CoreShapeFamily coreShapeFamily);

// Everything the per-harmonic all-pairs sum needs. All vectors are borrowed
// (owned by the caller for the duration of solve()); nothing here is copied.
struct FieldSolveHarmonicInputs {
    // Mesh points for this harmonic (from CoilMesher). Positions are shared across
    // harmonics; only the per-point `value` (current) differs per harmonic.
    const std::vector<FieldPoint>& inducingPoints;   // inducingFields[h].get_data()
    const std::vector<FieldPoint>& inducedPoints;    // inducedFields[h].get_data()

    // windingIndex per inducing point, hoisted out of the hot loop (it depends only
    // on the inducing point's turn). std::nullopt for fringing/gap points with no
    // turn. Same length as inducingPoints.
    const std::vector<std::optional<size_t>>& windingIndexPerInducingPoint;

    // Optional per-induced-point seed (real, imag) that the sum starts from — used
    // to fold in ROSHEN/SULLIVAN gap fringing already computed by the caller. Empty
    // means "start from zero". Same length as inducedPoints when present.
    const std::vector<std::pair<double, double>>& inducedSeed;

    // Core geometry for the induced-point in-core rejection test.
    double coreColumnWidth;
    double coreWidth;
    CoreShapeFamily coreShapeFamily;
};

class FieldSolverBackend {
public:
    virtual ~FieldSolverBackend() = default;

    // Compute the complex field at every induced point for one harmonic. `model` is
    // the configured per-pair field model (LAMMERANER / BINNS / WANG); it is called
    // read-only and may be shared across threads. Returns one ComplexFieldPoint per
    // induced point, in inducedPoints order (point / turn_index / label re-attached),
    // or std::nullopt if this backend cannot handle the inputs (caller → CPU).
    virtual std::optional<std::vector<ComplexFieldPoint>> solve(
        const FieldSolveHarmonicInputs& in, MagneticFieldStrengthModel& model) = 0;

    virtual std::string name() const = 0;
};

}  // namespace OpenMagnetics
