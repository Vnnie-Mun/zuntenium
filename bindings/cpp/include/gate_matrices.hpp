// gate_matrices.hpp — Standard quantum gate matrices as compile-time constants.
//
// All matrices are in the computational basis {|0⟩, |1⟩} with row-major layout.
// Single-qubit matrices: 2×2 = 4 elements
// Two-qubit matrices:    4×4 = 16 elements

#pragma once

#include <array>
#include <complex>
#include <cmath>
#include <numbers>   // C++20: std::numbers::pi, std::numbers::sqrt2

namespace zuntenium::gates {

using cdouble = std::complex<double>;
constexpr double PI    = 3.141592653589793238462643383279502884;
constexpr double SQRT2 = 1.414213562373095048801688724209698079;
constexpr double INV_SQRT2 = 1.0 / SQRT2;

// ── Compile-time single-qubit matrices (2×2, row-major) ──────────────────────

/// Identity
inline constexpr std::array<cdouble, 4> I_matrix() {
    return {{ {1,0},{0,0}, {0,0},{1,0} }};
}

/// Pauli-X (NOT)
inline constexpr std::array<cdouble, 4> X_matrix() {
    return {{ {0,0},{1,0}, {1,0},{0,0} }};
}

/// Pauli-Y
inline std::array<cdouble, 4> Y_matrix() {
    return {{ {0,0},{0,-1}, {0,1},{0,0} }};
}

/// Pauli-Z
inline constexpr std::array<cdouble, 4> Z_matrix() {
    return {{ {1,0},{0,0}, {0,0},{-1,0} }};
}

/// Hadamard
inline std::array<cdouble, 4> H_matrix() {
    return {{
        {INV_SQRT2, 0}, { INV_SQRT2, 0},
        {INV_SQRT2, 0}, {-INV_SQRT2, 0}
    }};
}

/// S gate: diag(1, i)
inline constexpr std::array<cdouble, 4> S_matrix() {
    return {{ {1,0},{0,0}, {0,0},{0,1} }};
}

/// T gate: diag(1, e^{iπ/4})
inline std::array<cdouble, 4> T_matrix() {
    const double angle = PI / 4.0;
    return {{ {1,0},{0,0}, {0,0},{std::cos(angle), std::sin(angle)} }};
}

/// Rx(θ) = exp(-i θ/2 X)
inline std::array<cdouble, 4> Rx_matrix(double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {{ {c,0},{0,-s}, {0,-s},{c,0} }};
}

/// Ry(θ) = exp(-i θ/2 Y)
inline std::array<cdouble, 4> Ry_matrix(double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {{ {c,0},{-s,0}, {s,0},{c,0} }};
}

/// Rz(θ) = exp(-i θ/2 Z) = diag(e^{-iθ/2}, e^{iθ/2})
inline std::array<cdouble, 4> Rz_matrix(double theta) {
    const double half = theta / 2.0;
    return {{
        {std::cos(-half), std::sin(-half)}, {0,0},
        {0,0},                               {std::cos(half), std::sin(half)}
    }};
}

/// U3(θ, φ, λ) — general single-qubit unitary
inline std::array<cdouble, 4> U3_matrix(double theta, double phi, double lam) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {{
        {c, 0},
        {-std::cos(lam) * s, -std::sin(lam) * s},
        {std::cos(phi) * s,   std::sin(phi) * s},
        {std::cos(phi + lam) * c, std::sin(phi + lam) * c}
    }};
}

// ── Two-qubit gate matrices (4×4 = 16 elements, row-major) ────────────────────
// State ordering: |00⟩, |01⟩, |10⟩, |11⟩

/// CNOT (CX): flips target qubit when control = |1⟩
inline constexpr std::array<cdouble, 16> CX_matrix() {
    std::array<cdouble, 16> m{};
    // |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    m[0]  = {1,0}; m[5]  = {1,0};   // rows 0 and 1 unchanged
    m[11] = {1,0}; m[14] = {1,0};   // rows 2 and 3 swapped
    return m;
}

/// CZ: applies Z to target when control = |1⟩
inline constexpr std::array<cdouble, 16> CZ_matrix() {
    std::array<cdouble, 16> m{};
    m[0] = m[5] = m[10] = {1,0};
    m[15] = {-1,0};
    return m;
}

/// SWAP
inline constexpr std::array<cdouble, 16> SWAP_matrix() {
    std::array<cdouble, 16> m{};
    m[0] = {1,0}; m[6] = {1,0}; m[9] = {1,0}; m[15] = {1,0};
    return m;
}

} // namespace zuntenium::gates
