#pragma once
// Minimal stand-in for MKF's support/Exceptions.h, so the backend can be compiled
// and unit-tested in isolation from the (currently mid-migration) MKF/MAS headers.
#include <stdexcept>
#include <string>

namespace OpenMagnetics {
class NaNResultException : public std::runtime_error {
public:
    explicit NaNResultException(const std::string& message) : std::runtime_error(message) {}
};
}  // namespace OpenMagnetics
