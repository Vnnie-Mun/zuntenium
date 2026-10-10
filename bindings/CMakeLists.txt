# CMakeLists.txt — ZUNTENIUM pybind11 C++ extension.
#
# Key fix: pybind11 is installed as a Python package (pip install pybind11),
# so we resolve its cmake dir via Python at configure time instead of relying
# on CMake's system search path.
#
# Usage:
#   cd /path/to/zuntenium
#   PYBIND11_DIR=$(python -m pybind11 --cmakedir)
#   mkdir -p build/_cmake && cd build/_cmake
#   cmake ../../bindings -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$PYBIND11_DIR
#   make -j$(nproc)
#   cp _zuntenium_core*.so ../../zuntenium/

cmake_minimum_required(VERSION 3.18)
project(zuntenium_bindings CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ─────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ── Compiler flags ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

# ── Python ────────────────────────────────────────────────────────────────────
# Find the Python executable and dev headers.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python version  : ${Python_VERSION}")
message(STATUS "Python includes : ${Python_INCLUDE_DIRS}")

# ── pybind11 via Python ────────────────────────────────────────────────────────
# pybind11 is pip-installed. We resolve its cmake dir via the Python interpreter
# and directly include() its tools — this avoids find_package() which the IDE
# CMake language server cannot resolve (it doesn't run execute_process).
set(PYBIND11_FOUND FALSE)

if(NOT DEFINED PYBIND11_CMAKE_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
    ERROR_QUIET
  )
  if(NOT PYBIND11_RESULT EQUAL 0 OR NOT PYBIND11_CMAKE_DIR)
    message(STATUS
      "pybind11 not found via Python. "
      "Install it with: pip install pybind11  "
      "Or pass: -DPYBIND11_CMAKE_DIR=/path/to/pybind11/share/cmake/pybind11"
    )
    set(PYBIND11_CMAKE_DIR "")
  endif()
endif()

# Directly include pybind11's cmake config — no find_package() needed.
if(PYBIND11_CMAKE_DIR)
  set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
  include("${PYBIND11_CMAKE_DIR}/pybind11Config.cmake" OPTIONAL RESULT_VARIABLE _pybind11_included)
  if(_pybind11_included)
    set(PYBIND11_FOUND TRUE)
    message(STATUS "pybind11 cmake dir: ${PYBIND11_CMAKE_DIR}")
    message(STATUS "pybind11 version : ${pybind11_VERSION}")
  else()
    message(STATUS "pybind11Config.cmake not found at ${PYBIND11_CMAKE_DIR} — skipping C++ bindings")
  endif()
else()
  message(STATUS "pybind11 not available — C++ bindings will not be built (pure-Python fallback)")
endif()

# ── Optional: BLAS / LAPACK ────────────────────────────────────────────────────
find_package(BLAS QUIET)
if(BLAS_FOUND)
  message(STATUS "BLAS found — linking for accelerated linear algebra")
else()
  message(STATUS "BLAS not found — using plain NumPy-equivalent C++ ops")
endif()

# ── Optional: OpenMP ──────────────────────────────────────────────────────────
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found — parallel gate application enabled")
endif()

# ── Build extension module (only when pybind11 is available) ───────────────────
if(PYBIND11_FOUND)
  # CMakeLists.txt lives inside bindings/ so paths are relative to that dir.
  set(BINDING_SOURCES
      "${CMAKE_CURRENT_SOURCE_DIR}/cpp/statevector_bindings.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/cpp/gate_matrix_bindings.cpp"
  )

  pybind11_add_module(_zuntenium_core MODULE ${BINDING_SOURCES})

  target_include_directories(_zuntenium_core PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
  )

  # Link optional libraries
  if(BLAS_FOUND)
    target_link_libraries(_zuntenium_core PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(_zuntenium_core PRIVATE ZUNTENIUM_USE_BLAS=1)
  endif()

  if(OpenMP_CXX_FOUND)
    target_link_libraries(_zuntenium_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_zuntenium_core PRIVATE ZUNTENIUM_USE_OMP=1)
  endif()

  set_target_properties(_zuntenium_core PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CXX_VISIBILITY_PRESET "hidden"
      VISIBILITY_INLINES_HIDDEN ON
  )

  # ── Install: copy .so into the Python package ────────────────────────────────
  cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH PROJECT_ROOT)

  install(TARGETS _zuntenium_core
      LIBRARY DESTINATION "${PROJECT_ROOT}/zuntenium"
      RUNTIME DESTINATION "${PROJECT_ROOT}/zuntenium"
  )

  # ── Strip release binaries ───────────────────────────────────────────────────
  if(CMAKE_BUILD_TYPE STREQUAL "Release" AND UNIX)
    find_program(STRIP_PROGRAM strip)
    if(STRIP_PROGRAM)
      add_custom_command(TARGET _zuntenium_core POST_BUILD
        COMMAND ${STRIP_PROGRAM} --strip-unneeded $<TARGET_FILE:_zuntenium_core>
        COMMENT "Stripping debug symbols"
      )
    endif()
  endif()

  message(STATUS "ZUNTENIUM bindings configured (${CMAKE_BUILD_TYPE})")
else()
  message(STATUS "ZUNTENIUM bindings skipped — install pybind11 to enable C++ acceleration")
endif()
