// pybind_module.cpp — pybind11 module definition for bindings_core.
//
// This is the reference pybind module in bindings_core/cpp/src/.
// The production build uses bindings/cpp/statevector_bindings.cpp instead,
// which has the same API but references the inline headers.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "statevector.hpp"

namespace py = pybind11;
using namespace zuntenium;

PYBIND11_MODULE(_zuntenium_cpp, m) {
    m.doc() = "ZUNTENIUM C++ core (bindings_core reference build).";

    m.attr("__version__") = "0.1.0";

    m.def("is_cpp_available", []() { return true; });

    py::class_<StateVectorCPP>(m, "StateVectorCPP")
        .def(py::init<uint32_t>(), py::arg("num_qubits"))
        .def("apply_single_qubit_gate",
             &StateVectorCPP::apply_single_qubit_gate,
             py::arg("matrix"), py::arg("qubit_index"))
        .def("apply_two_qubit_gate",
             &StateVectorCPP::apply_two_qubit_gate,
             py::arg("matrix"), py::arg("qubit0"), py::arg("qubit1"))
        .def("get_probabilities", &StateVectorCPP::get_probabilities)
        .def("get_amplitudes", &StateVectorCPP::get_amplitudes)
        .def("sample_measurements", &StateVectorCPP::sample_measurements,
             py::arg("shots"), py::arg("seed") = 42)
        .def("num_qubits", &StateVectorCPP::num_qubits)
        .def("dim", &StateVectorCPP::dim)
        .def("reset", &StateVectorCPP::reset);
}
