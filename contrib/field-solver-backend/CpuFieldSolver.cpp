#include "CpuFieldSolver.h"

#include <cmath>
#include <exception>

#include "support/Exceptions.h"

namespace OpenMagnetics {

namespace {

// The per-induced-point body, lifted verbatim from the original inner loop in
// MagneticField::calculate_magnetic_field_strength_field. Pure with respect to
// shared state: it only reads `in` and calls the model read-only, so it is safe to
// run concurrently for distinct induced points. Throws NaNResultException on a NaN
// contribution, exactly as the original did.
ComplexFieldPoint computeInducedPoint(size_t d, const FieldSolveHarmonicInputs& in,
                                      MagneticFieldStrengthModel& model) {
    const FieldPoint& inducedFieldPoint = in.inducedPoints[d];

    double totalInducedFieldX = 0;
    double totalInducedFieldY = 0;
    if (!in.inducedSeed.empty()) {
        totalInducedFieldX = in.inducedSeed[d].first;
        totalInducedFieldY = in.inducedSeed[d].second;
    }

    const size_t inducingCount = in.inducingPoints.size();
    for (size_t i = 0; i < inducingCount; ++i) {
        const FieldPoint& inducingFieldPoint = in.inducingPoints[i];
        const std::optional<size_t>& windingIndex = in.windingIndexPerInducingPoint[i];

        if (inducingFieldPoint.get_turn_index()) {
            if (inducedFieldPoint.get_turn_index()) {
                if (inducedFieldPoint.get_turn_index().value() == inducingFieldPoint.get_turn_index().value()) {
                    continue;
                }
            }
            else if (is_inside_core(inducedFieldPoint, in.coreColumnWidth, in.coreWidth, in.coreShapeFamily)) {
                continue;
            }
        }

        auto complexFieldPoint =
            model.get_magnetic_field_strength_between_two_points(inducingFieldPoint, inducedFieldPoint, windingIndex);

        totalInducedFieldX += complexFieldPoint.get_real();
        totalInducedFieldY += complexFieldPoint.get_imaginary();
        if (std::isnan(complexFieldPoint.get_real())) {
            throw NaNResultException("NaN found in magnetic field calculation");
        }
        if (std::isnan(complexFieldPoint.get_imaginary())) {
            throw NaNResultException("NaN found in magnetic field calculation");
        }
    }

    ComplexFieldPoint out;
    out.set_point(inducedFieldPoint.get_point());
    out.set_real(totalInducedFieldX);
    out.set_imaginary(totalInducedFieldY);
    if (inducedFieldPoint.get_turn_index()) {
        out.set_turn_index(inducedFieldPoint.get_turn_index().value());
    }
    if (inducedFieldPoint.get_label()) {
        out.set_label(inducedFieldPoint.get_label().value());
    }
    return out;
}

}  // namespace

std::optional<std::vector<ComplexFieldPoint>> CpuFieldSolver::solve(
    const FieldSolveHarmonicInputs& in, MagneticFieldStrengthModel& model) {
    const size_t inducedCount = in.inducedPoints.size();
    std::vector<ComplexFieldPoint> results(inducedCount);

#ifdef _OPENMP
    // A thrown exception must not cross an OpenMP region boundary — capture the
    // first one and rethrow after the parallel loop, preserving the exact type and
    // message. NaN still aborts the calculation; only the throw is deferred to the
    // end of this harmonic's sweep, which is unobservable to the caller.
    std::exception_ptr firstError = nullptr;
    #pragma omp parallel for schedule(dynamic, 8) if (inducedCount > FIELD_SOLVE_PARALLEL_MIN)
    for (long long d = 0; d < static_cast<long long>(inducedCount); ++d) {
        if (firstError) {
            continue;
        }
        try {
            results[static_cast<size_t>(d)] = computeInducedPoint(static_cast<size_t>(d), in, model);
        }
        catch (...) {
            #pragma omp critical
            {
                if (!firstError) {
                    firstError = std::current_exception();
                }
            }
        }
    }
    if (firstError) {
        std::rethrow_exception(firstError);
    }
#else
    for (size_t d = 0; d < inducedCount; ++d) {
        results[d] = computeInducedPoint(d, in, model);
    }
#endif

    return results;
}

}  // namespace OpenMagnetics
