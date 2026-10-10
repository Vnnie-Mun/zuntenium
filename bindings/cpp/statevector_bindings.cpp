// statevector_bindings.cpp — pybind11 module init for _zuntenium_core.
//
// This is the ONLY PYBIND11_MODULE entry point in the project.
// Gate matrix functions are registered via register_gate_matrices()
// defined in gate_matrix_bindings.cpp.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "statevector_core.hpp"

namespace py = pybind11;
using namespace zuntenium;

// Forward declaration — implemented in gate_matrix_bindings.cpp
void register_gate_matrices(py::module_& m);

PYBIND11_MODULE(_zuntenium_core, m) {
    m.doc() = "ZUNTENIUM C++ statevector simulation engine (pybind11).";

    m.attr("__version__") = "0.1.0";
    m.attr("__backend__") = "cpp";

    m.def("is_cpp_available", []() { return true; },
          "Returns True — this is the compiled C++ extension.");

    // ── StateVectorCPP ─────────────────────────────────────────────────────
    py::class_<StateVectorCPP>(m, "StateVector",
        R"doc(
        High-performance C++ statevector simulator.

        Memory layout: amplitudes[i] = coefficient of |i⟩.
        Qubit 0 is the most significant bit (big-endian convention).

        Args:
            num_qubits: Number of qubits (1–30).

        Example::

            sv = StateVector(2)
            INV = 0.7071067811865476
            sv.apply_single_qubit_gate([INV, INV, INV, -INV], 0)
            print(sv.get_probabilities())   # [0.5, 0.5, 0.0, 0.0]
        )doc")

        .def(py::init<int>(), py::arg("num_qubits"),
             "Initialize to |0...0⟩.")

        .def("apply_single_qubit_gate",
             &StateVectorCPP::apply_single_qubit_gate,
             py::arg("matrix"), py::arg("qubit"),
             "Apply a 2×2 unitary (4 complex values, row-major) to qubit.")

        .def("apply_two_qubit_gate",
             &StateVectorCPP::apply_two_qubit_gate,
             py::arg("matrix"), py::arg("q0"), py::arg("q1"),
             "Apply a 4×4 unitary (16 complex values, row-major) to qubits (q0, q1).")

        .def("get_amplitudes",    &StateVectorCPP::get_amplitudes,
             "Return amplitude vector as list of complex.")
        .def("get_probabilities", &StateVectorCPP::get_probabilities,
             "Return |amplitude|² as list of float.")
        .def("sample_measurements", &StateVectorCPP::sample_measurements,
             py::arg("shots"), py::arg("seed") = 42ULL,
             "Sample basis state indices. Returns list of int, length = shots.")
        .def("num_qubits", &StateVectorCPP::num_qubits, "Number of qubits.")
        .def("dim",        &StateVectorCPP::dim,        "Hilbert space dim (2^n).")
        .def("reset",      &StateVectorCPP::reset,      "Reset to |0...0⟩.")
        .def("__repr__", [](const StateVectorCPP& sv) {
            return "<StateVector(cpp) num_qubits="
                   + std::to_string(sv.num_qubits()) + ">";
        });

    // ── Factory ────────────────────────────────────────────────────────────
    m.def("create_statevector",
          [](int n) { return StateVectorCPP(n); },
          py::arg("num_qubits"),
          "Create a C++ StateVector initialized to |0...0⟩.");

    // ── Gate matrices submodule ────────────────────────────────────────────
    register_gate_matrices(m);
}
