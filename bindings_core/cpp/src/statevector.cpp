// statevector.cpp — StateVectorCPP implementation.
//
// Reference implementation. The production build uses the inline functions
// in bindings/cpp/include/statevector_core.hpp directly.

#include "statevector.hpp"
#include "gate_ops.hpp"

#include <algorithm>
#include <cassert>
#include <random>
#include <stdexcept>

namespace zuntenium {

StateVectorCPP::StateVectorCPP(uint32_t num_qubits)
    : num_qubits_(num_qubits)
    , dim_(uint64_t(1) << num_qubits)
    , amplitudes_(dim_, cdouble(0.0, 0.0))
{
    if (num_qubits < 1 || num_qubits > 30)
        throw std::invalid_argument("num_qubits must be in [1, 30]");
    amplitudes_[0] = cdouble(1.0, 0.0);
}

void StateVectorCPP::apply_single_qubit_gate(
    const std::vector<cdouble>& matrix,
    uint32_t qubit_index)
{
    if (matrix.size() != 4)
        throw std::invalid_argument("Single-qubit gate needs 4 elements");
    apply_single_qubit_op(amplitudes_, num_qubits_, qubit_index, matrix.data());
}

void StateVectorCPP::apply_two_qubit_gate(
    const std::vector<cdouble>& matrix,
    uint32_t qubit0,
    uint32_t qubit1)
{
    if (matrix.size() != 16)
        throw std::invalid_argument("Two-qubit gate needs 16 elements");
    apply_two_qubit_op(amplitudes_, num_qubits_, qubit0, qubit1, matrix.data());
}

std::vector<double> StateVectorCPP::get_probabilities() const {
    std::vector<double> probs(dim_);
    for (uint64_t i = 0; i < dim_; ++i)
        probs[i] = std::norm(amplitudes_[i]);
    return probs;
}

std::vector<cdouble> StateVectorCPP::get_amplitudes() const {
    return amplitudes_;
}

std::vector<int> StateVectorCPP::sample_measurements(int shots, uint64_t seed) const {
    std::vector<double> probs = get_probabilities();
    double total = 0.0;
    for (auto p : probs) total += p;
    for (auto& p : probs) p /= total;

    std::vector<double> cdf(dim_);
    cdf[0] = probs[0];
    for (uint64_t i = 1; i < dim_; ++i)
        cdf[i] = cdf[i - 1] + probs[i];

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<int> results(shots);
    for (int s = 0; s < shots; ++s) {
        double r = dist(rng);
        auto it = std::lower_bound(cdf.begin(), cdf.end(), r);
        results[s] = static_cast<int>(std::min(
            it - cdf.begin(), static_cast<std::ptrdiff_t>(dim_ - 1)));
    }
    return results;
}

void StateVectorCPP::reset() {
    std::fill(amplitudes_.begin(), amplitudes_.end(), cdouble(0.0));
    amplitudes_[0] = cdouble(1.0, 0.0);
}

} // namespace zuntenium
