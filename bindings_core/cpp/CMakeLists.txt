# bindings_core/cpp — ZUNTENIUM Native C++ Simulation Core
#
# This is the canonical C++ source tree. The actual build is driven by
# bindings/CMakeLists.txt which references the headers in bindings/cpp/include/.
# This directory mirrors the architecture spec's bindings_core/cpp/ layout
# and contains the reference implementations.

cmake_minimum_required(VERSION 3.18)
project(zuntenium_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Headers
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Sources
set(SOURCES
    src/statevector.cpp
    src/gate_ops.cpp
    src/pybind_module.cpp
)

# For standalone build (testing only — production uses bindings/CMakeLists.txt)
find_package(Python COMPONENTS Interpreter Development.Module QUIET)
find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(_zuntenium_cpp ${SOURCES})
    target_include_directories(_zuntenium_cpp PRIVATE ${INCLUDE_DIR})
    message(STATUS "bindings_core/cpp: pybind11 build configured")
else()
    message(STATUS "bindings_core/cpp: pybind11 not found — skipping (use bindings/CMakeLists.txt)")
endif()
