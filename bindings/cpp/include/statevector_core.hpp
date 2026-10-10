// statevector_core.hpp — High-performance statevector simulation engine.
//
// All free functions are marked inline to prevent ODR violations when
// this header is included by multiple translation units.

#pragma once

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ZUNTENIUM_USE_OMP
#  include <omp.h>
#endif

namespace zuntenium {

using cdouble = std::complex<double>;

// ── Utilities ────────────────────────────────────────────────────────────────

inline std::size_t dim_of(int num_qubits) {
    return std::size_t(1) << num_qubits;
}

inline std::size_t qubit_mask(int q, int num_qubits) {
    return std::size_t(1) << (num_qubits - 1 - q);
}

// ── Single-qubit gate ────────────────────────────────────────────────────────

/// Apply a 2×2 unitary to qubit `q`. `matrix` is row-major [m00,m01,m10,m11].
inline void apply_single_qubit(
    cdouble* amplitudes,
    int num_qubits,
    int qubit,
    const cdouble matrix[4])
{
    const std::size_t n   = dim_of(num_qubits);
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

// ── Two-qubit gate ───────────────────────────────────────────────────────────

/// Apply a 4×4 unitary to qubits (q0, q1). `matrix` is row-major, 16 elems.
inline void apply_two_qubit(
    cdouble* amplitudes,
    int num_qubits,
    int q0,
    int q1,
    const cdouble matrix[16])
{
    const std::size_t n    = dim_of(num_qubits);
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

            amplitudes[i00] = matrix[0]*a00 + matrix[1]*a01  + matrix[2]*a10  + matrix[3]*a11;
            amplitudes[i01] = matrix[4]*a00 + matrix[5]*a01  + matrix[6]*a10  + matrix[7]*a11;
            amplitudes[i10] = matrix[8]*a00 + matrix[9]*a01  + matrix[10]*a10 + matrix[11]*a11;
            amplitudes[i11] = matrix[12]*a00 + matrix[13]*a01 + matrix[14]*a10 + matrix[15]*a11;
        }
    }
}

// ── Sampling ─────────────────────────────────────────────────────────────────

inline std::vector<int> sample(
    const cdouble* amplitudes,
    int num_qubits,
    int shots,
    uint64_t seed = 42)
{
    const std::size_t n = dim_of(num_qubits);
    std::vector<double> probs(n);
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::norm(amplitudes[i]);
        total   += probs[i];
    }
    for (auto& p : probs) p /= total;

    std::vector<double> cdf(n);
    cdf[0] = probs[0];
    for (std::size_t i = 1; i < n; ++i)
        cdf[i] = cdf[i - 1] + probs[i];

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<int> results(shots);
    for (int s = 0; s < shots; ++s) {
        double r  = dist(rng);
        auto   it = std::lower_bound(cdf.begin(), cdf.end(), r);
        results[s] = static_cast<int>(std::min(it - cdf.begin(),
                                               static_cast<std::ptrdiff_t>(n - 1)));
    }
    return results;
}

// ── StateVectorCPP ────────────────────────────────────────────────────────────

class StateVectorCPP {
public:
    explicit StateVectorCPP(int num_qubits)
        : _num_qubits(num_qubits)
        , _dim(dim_of(num_qubits))
        , _amplitudes(_dim, cdouble(0.0, 0.0))
    {
        if (num_qubits < 1 || num_qubits > 30)
            throw std::invalid_argument(
                "num_qubits must be in [1, 30], got " + std::to_string(num_qubits));
        _amplitudes[0] = cdouble(1.0, 0.0);
    }

    void apply_single_qubit_gate(const std::vector<cdouble>& mat, int qubit) {
        if (mat.size() != 4)
            throw std::invalid_argument("Single-qubit gate needs 4 elements");
        if (qubit < 0 || qubit >= _num_qubits)
            throw std::out_of_range("Qubit index out of range");
        apply_single_qubit(_amplitudes.data(), _num_qubits, qubit, mat.data());
    }

    void apply_two_qubit_gate(const std::vector<cdouble>& mat, int q0, int q1) {
        if (mat.size() != 16)
            throw std::invalid_argument("Two-qubit gate needs 16 elements");
        if (q0 < 0 || q0 >= _num_qubits || q1 < 0 || q1 >= _num_qubits)
            throw std::out_of_range("Qubit index out of range");
        if (q0 == q1)
            throw std::invalid_argument("q0 and q1 must differ");
        apply_two_qubit(_amplitudes.data(), _num_qubits, q0, q1, mat.data());
    }

    std::vector<cdouble> get_amplitudes() const { return _amplitudes; }

    std::vector<double> get_probabilities() const {
        std::vector<double> p(_dim);
        for (std::size_t i = 0; i < _dim; ++i)
            p[i] = std::norm(_amplitudes[i]);
        return p;
    }

    std::vector<int> sample_measurements(int shots, uint64_t seed = 42) const {
        return sample(_amplitudes.data(), _num_qubits, shots, seed);
    }

    int         num_qubits() const { return _num_qubits; }
    std::size_t dim()        const { return _dim; }

    void reset() {
        std::fill(_amplitudes.begin(), _amplitudes.end(), cdouble(0.0));
        _amplitudes[0] = cdouble(1.0, 0.0);
    }

private:
    int                  _num_qubits;
    std::size_t          _dim;
    std::vector<cdouble> _amplitudes;
};

} // namespace zuntenium
