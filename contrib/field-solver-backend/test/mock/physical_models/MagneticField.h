#pragma once
// Minimal stand-in for MKF's physical_models/MagneticField.h — just the surface the
// FieldSolverBackend touches — so CpuFieldSolver can be compiled and unit-tested
// without the full MKF/MAS header tree. Signatures mirror the real types exactly:
//   FieldPoint::get_turn_index() -> std::optional<size_t>
//   FieldPoint::get_point()      -> std::vector<double>
//   FieldPoint::get_label()      -> std::optional<std::string>
//   ComplexFieldPoint  get/set real, imaginary, point, turn_index, label
//   MagneticFieldStrengthModel::get_magnetic_field_strength_between_two_points(
//       FieldPoint inducing, FieldPoint induced, std::optional<size_t> wireIndex)
#include <optional>
#include <string>
#include <vector>

namespace OpenMagnetics {

enum class CoreShapeFamily { C, E, ETD, P, PM, PQ, RM, T, U };

class FieldPoint {
public:
    std::vector<double> get_point() const { return _point; }
    void set_point(std::vector<double> point) { _point = point; }
    std::optional<size_t> get_turn_index() const { return _turnIndex; }
    void set_turn_index(size_t turnIndex) { _turnIndex = turnIndex; }
    std::optional<std::string> get_label() const { return _label; }
    void set_label(std::string label) { _label = label; }
    double get_value() const { return _value; }
    void set_value(double value) { _value = value; }

private:
    std::vector<double> _point;
    std::optional<size_t> _turnIndex;
    std::optional<std::string> _label;
    double _value = 0.0;
};

class ComplexFieldPoint {
public:
    double get_real() const { return _real; }
    void set_real(double real) { _real = real; }
    double get_imaginary() const { return _imaginary; }
    void set_imaginary(double imaginary) { _imaginary = imaginary; }
    std::vector<double> get_point() const { return _point; }
    void set_point(std::vector<double> point) { _point = point; }
    std::optional<size_t> get_turn_index() const { return _turnIndex; }
    void set_turn_index(size_t turnIndex) { _turnIndex = turnIndex; }
    std::optional<std::string> get_label() const { return _label; }
    void set_label(std::string label) { _label = label; }

private:
    double _real = 0.0;
    double _imaginary = 0.0;
    std::vector<double> _point;
    std::optional<size_t> _turnIndex;
    std::optional<std::string> _label;
};

class MagneticFieldStrengthModel {
public:
    virtual ~MagneticFieldStrengthModel() = default;
    virtual ComplexFieldPoint get_magnetic_field_strength_between_two_points(
        FieldPoint inducingFieldPoint, FieldPoint inducedFieldPoint,
        std::optional<size_t> inducingWireIndex = std::nullopt) = 0;
};

}  // namespace OpenMagnetics
