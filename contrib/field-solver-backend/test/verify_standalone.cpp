// Standalone unit test for CpuFieldSolver, compiled against the mock MKF types in
// test/mock (the real MKF/MAS headers are mid-migration on this checkout and don't
// compile). Proves the extracted loop is valid C++ and logically correct, and that
// the OpenMP path is bit-identical to serial.
//
// Build serial : g++ -std=c++23 -I test/mock -I . test/verify_standalone.cpp CpuFieldSolver.cpp -o v && ./v
// Build OpenMP : add -fopenmp  (same asserts must pass; result must match serial)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "CpuFieldSolver.h"

using namespace OpenMagnetics;

static constexpr double PI = 3.14159265358979323846;

// --- the free function the backend calls; real one lives in MagneticField.cpp -----
// For the test: an induced point is "inside the core" iff its x-coordinate < 0.
namespace OpenMagnetics {
bool is_inside_core(FieldPoint inducedFieldPoint, double /*coreColumnWidth*/,
                    double /*coreWidth*/, CoreShapeFamily /*coreShapeFamily*/) {
    return inducedFieldPoint.get_point()[0] < 0.0;
}
}  // namespace OpenMagnetics

// --- deterministic per-pair model (LAMMERANER-shaped; wireIndex folded in so the
//     hoisted windingIndex threading is verified) -----------------------------------
class MockModel : public MagneticFieldStrengthModel {
public:
    ComplexFieldPoint get_magnetic_field_strength_between_two_points(
        FieldPoint inducing, FieldPoint induced, std::optional<size_t> wireIndex) override {
        auto ip = inducing.get_point();
        auto dp = induced.get_point();
        double ix = ip[0], iy = ip[1], xd = dp[0], yd = dp[1];
        double val = inducing.get_value();
        double wfac = wireIndex.has_value() ? (1.0 + 0.01 * static_cast<double>(wireIndex.value())) : 1.0;
        double dx = xd - ix, dy = yd - iy;
        double dist = std::sqrt(dx * dx + dy * dy);
        ComplexFieldPoint c;
        if (dist < 1e-12) { c.set_real(0.0); c.set_imaginary(0.0); return c; }
        double tl = 1.0;
        double m = -val / (2.0 * PI * dist) * tl / std::sqrt(tl * tl + dist * dist) * wfac;
        c.set_real(m * (dx / dist));
        c.set_imaginary(m * (-dy / dist));
        return c;
    }
};

int main() {
    MockModel model;

    // --- synthetic inputs -------------------------------------------------------
    // Inducing points: 3 real turns (turnIndex 0,1,2) each with a value, winding
    // lookup 0/0/1; plus one gap/fringing point (no turnIndex).
    std::vector<FieldPoint> inducing;
    auto mkInd = [](double x, double y, double val, std::optional<size_t> turn) {
        FieldPoint p; p.set_point({x, y}); p.set_value(val);
        if (turn) p.set_turn_index(turn.value());
        return p;
    };
    // Positions are offset from the induced turn centres (in reality the inducing
    // points are mirror images), so a same-turn self-pair would contribute non-zero
    // if it were not skipped — which lets the skip-self assert below actually bite.
    inducing.push_back(mkInd(0.0105, 0.0002, 4.0, 0));
    inducing.push_back(mkInd(0.0110, 0.0050, 4.0, 1));
    inducing.push_back(mkInd(0.0122, -0.0038, 1.3, 2));
    inducing.push_back(mkInd(0.0000, 0.0200, 2.0, std::nullopt));   // gap point, no turn
    std::vector<std::optional<size_t>> windingIndex = {0, 0, 1, std::nullopt};

    // Induced points: a couple of turn centres, a leakage point in-core (x<0), and
    // a leakage point outside the core (x>0).
    std::vector<FieldPoint> induced;
    auto mkDd = [](double x, double y, std::optional<size_t> turn, std::optional<std::string> label) {
        FieldPoint p; p.set_point({x, y});
        if (turn) p.set_turn_index(turn.value());
        if (label) p.set_label(label.value());
        return p;
    };
    induced.push_back(mkDd(0.010, 0.000, 0, std::nullopt));           // turn 0 centre (self-pair with inducing[0])
    induced.push_back(mkDd(0.012, -0.004, 2, std::nullopt));          // turn 2 centre (self-pair with inducing[2])
    induced.push_back(mkDd(-0.003, 0.001, std::nullopt, std::string("leak_in_core")));   // in core (x<0)
    induced.push_back(mkDd(0.020, 0.002, std::nullopt, std::string("leak_outside")));     // outside core

    // Fringing seed (ROSHEN/SULLIVAN-style) for two of the induced points.
    std::vector<std::pair<double, double>> seed(induced.size(), {0.0, 0.0});
    seed[2] = {7.5, -2.5};
    seed[3] = {0.25, 0.125};

    FieldSolveHarmonicInputs in{inducing, induced, windingIndex, seed, 0.006, 0.02, CoreShapeFamily::U};

    // --- run the backend --------------------------------------------------------
    CpuFieldSolver solver;
    auto solvedOpt = solver.solve(in, model);
    assert(solvedOpt.has_value());
    auto& got = solvedOpt.value();
    assert(got.size() == induced.size());

    // --- independent reference (different structure: build contributor set, then
    //     reduce in the same index order so serial FP result is bit-comparable) ----
    auto reference = [&](size_t d) {
        double rx = seed[d].first, ry = seed[d].second;
        const auto& dd = induced[d];
        for (size_t i = 0; i < inducing.size(); ++i) {
            const auto& ind = inducing[i];
            bool skip = false;
            if (ind.get_turn_index()) {
                if (dd.get_turn_index()) {
                    if (dd.get_turn_index().value() == ind.get_turn_index().value()) skip = true;
                } else if (is_inside_core(dd, in.coreColumnWidth, in.coreWidth, in.coreShapeFamily)) {
                    skip = true;
                }
            }
            if (skip) continue;
            auto c = model.get_magnetic_field_strength_between_two_points(ind, dd, windingIndex[i]);
            rx += c.get_real();
            ry += c.get_imaginary();
        }
        return std::pair<double, double>{rx, ry};
    };

    int failures = 0;
    for (size_t d = 0; d < induced.size(); ++d) {
        auto [rx, ry] = reference(d);
        if (got[d].get_real() != rx || got[d].get_imaginary() != ry) {
            std::printf("  point %zu MISMATCH: got (%.17g,%.17g) ref (%.17g,%.17g)\n",
                        d, got[d].get_real(), got[d].get_imaginary(), rx, ry);
            ++failures;
        }
    }

    // --- structural assertions --------------------------------------------------
    // turn_index propagated where present; label propagated where present.
    assert(got[0].get_turn_index().has_value() && got[0].get_turn_index().value() == 0);
    assert(!got[2].get_turn_index().has_value());
    assert(got[2].get_label().has_value() && got[2].get_label().value() == "leak_in_core");
    assert(got[0].get_point().size() == 2 && got[0].get_point()[0] == 0.010);

    // seed applied: point 3 (outside core) must include its seed plus all 4 inducing
    // contributions (nothing skipped — dd has no turn, and it's not in core, so even
    // the turn-bearing inducing points contribute).
    {
        auto [rx3, ry3] = reference(3);
        assert(got[3].get_real() == rx3 && got[3].get_imaginary() == ry3);
        // and the seed genuinely contributed: solver value minus the no-seed sum
        // equals the seed (0.25) exactly.
        double noSeedReal = rx3 - 0.25;
        assert(std::abs((got[3].get_real() - noSeedReal) - 0.25) < 1e-12);
    }

    // skip-self really happened: point 0 excluded inducing[0] (same turn 0).
    {
        // recompute point 0 WITHOUT the skip and confirm it differs → skip was active.
        double rx = seed[0].first, ry = seed[0].second;
        for (size_t i = 0; i < inducing.size(); ++i) {
            auto c = model.get_magnetic_field_strength_between_two_points(inducing[i], induced[0], windingIndex[i]);
            rx += c.get_real(); ry += c.get_imaginary();
        }
        assert(got[0].get_real() != rx);  // includes-self differs from solver's skip-self
    }

#ifdef _OPENMP
    std::printf("[OpenMP build] threads active; result compared against serial reference\n");
#endif

    if (failures == 0) {
        std::printf("PASS  %zu induced points, serial==reference bit-exact; skip-self, in-core, seed, propagation verified\n",
                    induced.size());
        return 0;
    }
    std::printf("FAIL  %d mismatches\n", failures);
    return 1;
}
