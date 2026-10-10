// gate_matrix_bindings.cpp — Gate matrix registration helper.
//
// Does NOT define PYBIND11_MODULE — there can only be one per .so.
// Instead, exposes `register_gate_matrices(py::module_& m)` which is called
// from statevector_bindings.cpp's PYBIND11_MODULE block.

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gate_matrices.hpp"

namespace py = pybind11;
using namespace zuntenium::gates;
using cdouble = std::complex<double>;

template <std::size_t N>
static std::vector<cdouble> to_vec(const std::array<cdouble, N> &arr) {
  return std::vector<cdouble>(arr.begin(), arr.end());
}

// Called from statevector_bindings.cpp to attach gate matrix functions
void register_gate_matrices(py::module_ &m) {

  py::module_ g = m.def_submodule(
      "gates",
      "Standard quantum gate matrices as flat lists of complex numbers.\n\n"
      "All single-qubit matrices: 4 elements (2×2 row-major).\n"
      "All two-qubit  matrices: 16 elements (4×4 row-major).");

  // ── Single-qubit ────────────────────────────────────────────────────────
  g.def("i_matrix", []() { return to_vec(I_matrix()); }, "2×2 identity.");
  g.def("x_matrix", []() { return to_vec(X_matrix()); }, "Pauli-X (NOT).");
  g.def("y_matrix", []() { return to_vec(Y_matrix()); }, "Pauli-Y.");
  g.def("z_matrix", []() { return to_vec(Z_matrix()); }, "Pauli-Z.");
  g.def("h_matrix", []() { return to_vec(H_matrix()); }, "Hadamard.");
  g.def("s_matrix", []() { return to_vec(S_matrix()); }, "S gate.");
  g.def("t_matrix", []() { return to_vec(T_matrix()); }, "T gate.");

  g.def(
      "rx_matrix", [](double theta) { return to_vec(Rx_matrix(theta)); },
      py::arg("theta"), "Rx(θ) = exp(-iθ/2 X).");
  g.def(
      "ry_matrix", [](double theta) { return to_vec(Ry_matrix(theta)); },
      py::arg("theta"), "Ry(θ) = exp(-iθ/2 Y).");
  g.def(
      "rz_matrix", [](double theta) { return to_vec(Rz_matrix(theta)); },
      py::arg("theta"), "Rz(θ) = exp(-iθ/2 Z).");
  g.def(
      "u3_matrix",
      [](double theta, double phi, double lam) {
        return to_vec(U3_matrix(theta, phi, lam));
      },
      py::arg("theta"), py::arg("phi"), py::arg("lam"),
      "U3(θ,φ,λ) — general single-qubit unitary.");

  // ── Two-qubit ────────────────────────────────────────────────────────────
  g.def("cx_matrix", []() { return to_vec(CX_matrix()); }, "CNOT.");
  g.def("cz_matrix", []() { return to_vec(CZ_matrix()); }, "Controlled-Z.");
  g.def("swap_matrix", []() { return to_vec(SWAP_matrix()); }, "SWAP.");
}
