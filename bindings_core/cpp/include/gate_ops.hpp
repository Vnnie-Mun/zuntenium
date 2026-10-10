// gate_ops.hpp — Gate operation kernels.
//
// Provides optimised gate application routines:
// - Single-qubit gate (2×2 unitary)
// - Two-qubit gate (4×4 unitary)
// - Future: SIMD intrinsics, OpenMP parallelism, cuBLAS GPU offload

#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace zuntenium {

using cdouble = std::complex<double>;

/// Apply a 2×2 unitary to a single qubit within the statevector.
void apply_single_qubit_op(
    std::vector<cdouble>& amplitudes,
    uint32_t num_qubits,
    uint32_t qubit,
    const cdouble matrix[4]);

/// Apply a 4×4 unitary to a pair of qubits within the statevector.
void apply_two_qubit_op(
    std::vector<cdouble>& amplitudes,
    uint32_t num_qubits,
    uint32_t q0,
    uint32_t q1,
    const cdouble matrix[16]);

} // namespace zuntenium
