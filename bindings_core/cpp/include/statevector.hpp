// statevector.hpp — Core statevector simulation engine.
//
// This is the reference header in bindings_core/cpp/.
// The production build uses the copy at bindings/cpp/include/statevector_core.hpp.
// See that file for the full inline implementation.

#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <string>

namespace zuntenium {

using cdouble = std::complex<double>;

class StateVectorCPP {
public:
    explicit StateVectorCPP(uint32_t num_qubits);

    void apply_single_qubit_gate(
        const std::vector<cdouble>& matrix,
        uint32_t qubit_index);

    void apply_two_qubit_gate(
        const std::vector<cdouble>& matrix,
        uint32_t qubit0,
        uint32_t qubit1);

    std::vector<double> get_probabilities() const;
    std::vector<cdouble> get_amplitudes() const;
    std::vector<int> sample_measurements(int shots, uint64_t seed = 42) const;

    uint32_t num_qubits() const { return num_qubits_; }
    uint64_t dim() const { return dim_; }
    void reset();

private:
    uint32_t num_qubits_;
    uint64_t dim_;
    std::vector<cdouble> amplitudes_;
};

} // namespace zuntenium
