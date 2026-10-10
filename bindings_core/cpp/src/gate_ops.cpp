// gate_ops.cpp — Gate application kernels.

#include "gate_ops.hpp"
#include <cstddef>

namespace zuntenium {

static std::size_t qubit_mask(uint32_t q, uint32_t num_qubits) {
    return std::size_t(1) << (num_qubits - 1 - q);
}

void apply_single_qubit_op(
    std::vector<cdouble>& amplitudes,
    uint32_t num_qubits,
    uint32_t qubit,
    const cdouble matrix[4])
{
    const std::size_t n = std::size_t(1) << num_qubits;
    const std::size_t bit = qubit_mask(qubit, num_qubits);

    const cdouble m00 = matrix[0], m01 = matrix[1];
    const cdouble m10 = matrix[2], m11 = matrix[3];

    for (std::size_t i = 0; i < n; i += 2 * bit) {
        for (std::size_t j = i; j < i + bit; ++j) {
            const cdouble a0 = amplitudes[j];
            const cdouble a1 = amplitudes[j + bit];
            amplitudes[j]       = m00 * a0 + m01 * a1;
            amplitudes[j + bit] = m10 * a0 + m11 * a1;
        }
    }
}

void apply_two_qubit_op(
    std::vector<cdouble>& amplitudes,
    uint32_t num_qubits,
    uint32_t q0,
    uint32_t q1,
    const cdouble matrix[16])
{
    const std::size_t n = std::size_t(1) << num_qubits;
    const std::size_t bit0 = qubit_mask(q0, num_qubits);
    const std::size_t bit1 = qubit_mask(q1, num_qubits);

    for (std::size_t i = 0; i < n; ++i) {
        if ((i & bit0) == 0 && (i & bit1) == 0) {
            const std::size_t i00 = i;
            const std::size_t i01 = i | bit1;
            const std::size_t i10 = i | bit0;
            const std::size_t i11 = i | bit0 | bit1;

            const cdouble a00 = amplitudes[i00];
            const cdouble a01 = amplitudes[i01];
            const cdouble a10 = amplitudes[i10];
            const cdouble a11 = amplitudes[i11];

            amplitudes[i00] = matrix[0]*a00  + matrix[1]*a01  + matrix[2]*a10  + matrix[3]*a11;
            amplitudes[i01] = matrix[4]*a00  + matrix[5]*a01  + matrix[6]*a10  + matrix[7]*a11;
            amplitudes[i10] = matrix[8]*a00  + matrix[9]*a01  + matrix[10]*a10 + matrix[11]*a11;
            amplitudes[i11] = matrix[12]*a00 + matrix[13]*a01 + matrix[14]*a10 + matrix[15]*a11;
        }
    }
}

} // namespace zuntenium
