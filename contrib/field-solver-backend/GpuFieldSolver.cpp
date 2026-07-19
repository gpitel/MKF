#include "GpuFieldSolver.h"

#include <cmath>

#ifdef __EMSCRIPTEN__
#include <emscripten/val.h>
#endif

namespace OpenMagnetics {

namespace {

// GPU-eligible iff every winding's wire is round or litz. A non-round wire (rect /
// FOIL / planar) makes both LAMMERANER and BINNS delegate to the BINNS-rectangular
// path per pair, which the WGSL kernel does not implement.
bool all_wires_round(const MagneticFieldStrengthModel& model) {
    if (model._wirePerWinding.empty()) {
        return false;
    }
    for (const auto& wire : model._wirePerWinding) {
        auto type = wire.get_type();
        if (type != WireType::ROUND && type != WireType::LITZ) {
            return false;
        }
    }
    return true;
}

// -1 => Lammeraner, 1 => Binns, 0 => unsupported. Mirrors the WGSL `model` field
// (0=Lammeraner, 1=Binns); returned here as an int with 2 == "unsupported".
int model_id(MagneticFieldStrengthModel& model) {
    if (dynamic_cast<MagneticFieldStrengthLammeranerModel*>(&model)) {
        return 0;
    }
    if (dynamic_cast<MagneticFieldStrengthBinnsLawrensonModel*>(&model)) {
        return 1;
    }
    return 2;
}

}  // namespace

std::shared_ptr<GpuFieldSolver> GpuFieldSolver::create() {
#ifdef __EMSCRIPTEN__
    // The JS side (mkfWorker.js) installs Module.webgpuFieldSolve once a WebGPU
    // adapter has been acquired; if it is absent there is no usable GPU.
    emscripten::val module = emscripten::val::global("Module");
    if (module.isUndefined() || module["webgpuFieldSolve"].isUndefined()) {
        return nullptr;
    }
    return std::make_shared<GpuFieldSolver>();
#else
    return nullptr;  // native build: no WebGPU path, always fall back to CPU
#endif
}

std::optional<std::vector<ComplexFieldPoint>> GpuFieldSolver::solve(
    const FieldSolveHarmonicInputs& in, MagneticFieldStrengthModel& model) {
    const int modelId = model_id(model);
    if (modelId > 1 || !all_wires_round(model)) {
        return std::nullopt;  // unsupported model or non-round wire → CPU fallback
    }
    if (in.inducedPoints.size() <= FIELD_SOLVE_PARALLEL_MIN) {
        return std::nullopt;  // too small to beat the ~15 ms GPU dispatch floor → CPU
    }

    const size_t inducingCount = in.inducingPoints.size();
    const size_t inducedCount = in.inducedPoints.size();
    const size_t windingCount = model._wirePerWinding.size();

    // Pack the borrowed MKF vectors into the flat SoA that webgpuFieldSolver.js
    // consumes (single harmonic per call — this interface is invoked once per
    // harmonic). Field semantics mirror the LAMMERANER/BINNS per-pair models.
    [[maybe_unused]] std::vector<float> indX(inducingCount), indY(inducingCount), indTurnLength(inducingCount), indValue(inducingCount);
    [[maybe_unused]] std::vector<int32_t> indTurnIndex(inducingCount), indWireIndex(inducingCount);
    for (size_t i = 0; i < inducingCount; ++i) {
        const auto& p = in.inducingPoints[i];
        const auto pt = p.get_point();
        indX[i] = static_cast<float>(pt[0]);
        indY[i] = static_cast<float>(pt[1]);
        indTurnLength[i] = static_cast<float>(p.get_turn_length() ? p.get_turn_length().value() : 1.0);
        indValue[i] = static_cast<float>(p.get_value());
        indTurnIndex[i] = p.get_turn_index() ? static_cast<int32_t>(p.get_turn_index().value()) : -1;
        indWireIndex[i] = in.windingIndexPerInducingPoint[i]
                              ? static_cast<int32_t>(in.windingIndexPerInducingPoint[i].value())
                              : -1;
    }

    [[maybe_unused]] std::vector<float> ddX(inducedCount), ddY(inducedCount);
    [[maybe_unused]] std::vector<int32_t> ddTurnIndex(inducedCount);
    [[maybe_unused]] std::vector<uint32_t> ddInsideCore(inducedCount);
    for (size_t d = 0; d < inducedCount; ++d) {
        const auto& p = in.inducedPoints[d];
        const auto pt = p.get_point();
        ddX[d] = static_cast<float>(pt[0]);
        ddY[d] = static_cast<float>(pt[1]);
        ddTurnIndex[d] = p.get_turn_index() ? static_cast<int32_t>(p.get_turn_index().value()) : -1;
        ddInsideCore[d] = is_inside_core(p, in.coreColumnWidth, in.coreWidth, in.coreShapeFamily) ? 1u : 0u;
    }

    // Per-winding wire radius / round-flag (radius = maxOuterWidth/2, as the models use).
    [[maybe_unused]] std::vector<float> wireRadius(windingCount), wireIsRound(windingCount);
    for (size_t w = 0; w < windingCount; ++w) {
        wireRadius[w] = static_cast<float>(model._wireMaxOuterWidth[w] / 2.0);
        auto type = model._wirePerWinding[w].get_type();
        wireIsRound[w] = (type == WireType::ROUND || type == WireType::LITZ) ? 1.0f : 0.0f;
    }

#ifdef __EMSCRIPTEN__
    using emscripten::val;
    auto toF32 = [](const std::vector<float>& v) {
        return val(emscripten::typed_memory_view(v.size(), v.data()));
    };
    auto toI32 = [](const std::vector<int32_t>& v) {
        return val(emscripten::typed_memory_view(v.size(), v.data()));
    };
    auto toU32 = [](const std::vector<uint32_t>& v) {
        return val(emscripten::typed_memory_view(v.size(), v.data()));
    };

    val inp = val::object();
    inp.set("Ni", static_cast<unsigned>(inducingCount));
    inp.set("Nd", static_cast<unsigned>(inducedCount));
    inp.set("harmonics", 1u);
    inp.set("model", modelId == 1 ? std::string("binns") : std::string("lammeraner"));
    inp.set("indX", toF32(indX));
    inp.set("indY", toF32(indY));
    inp.set("indTurnLength", toF32(indTurnLength));
    inp.set("indValue", toF32(indValue));
    inp.set("indTurnIndex", toI32(indTurnIndex));
    inp.set("indWireIndex", toI32(indWireIndex));
    inp.set("ddX", toF32(ddX));
    inp.set("ddY", toF32(ddY));
    inp.set("ddTurnIndex", toI32(ddTurnIndex));
    inp.set("ddInsideCore", toU32(ddInsideCore));
    inp.set("wireRadius", toF32(wireRadius));
    inp.set("wireIsRound", toF32(wireIsRound));

    // Async WebGPU solve; .await() requires the build to enable -sASYNCIFY. A null
    // return from JS (e.g. unsupported model slipped through) → CPU fallback.
    val result = val::global("Module")["webgpuFieldSolve"](inp).await();
    if (result.isNull() || result.isUndefined()) {
        return std::nullopt;
    }
    val realArr = result["real"];
    val imagArr = result["imag"];

    std::vector<ComplexFieldPoint> out(inducedCount);
    for (size_t d = 0; d < inducedCount; ++d) {
        double re = realArr[d].as<double>();
        double im = imagArr[d].as<double>();
        if (!in.inducedSeed.empty()) {          // fold in ROSHEN/SULLIVAN gap fringing
            re += in.inducedSeed[d].first;
            im += in.inducedSeed[d].second;
        }
        const auto& p = in.inducedPoints[d];
        out[d].set_point(p.get_point());
        out[d].set_real(re);
        out[d].set_imaginary(im);
        if (p.get_turn_index()) {
            out[d].set_turn_index(p.get_turn_index().value());
        }
        if (p.get_label()) {
            out[d].set_label(p.get_label().value());
        }
    }
    return out;
#else
    return std::nullopt;  // native: no WebGPU dispatch
#endif
}

}  // namespace OpenMagnetics
