// circuit_exec.hpp — Circuit execution engine (future).
//
// Will provide a C++ circuit executor that processes a serialised IR
// representation directly, avoiding Python overhead for large circuits.
//
// v0.1: Header-only stub.
// v0.2: Full implementation with IR deserialization.

#pragma once

#include <string>
#include <vector>
#include <map>

namespace zuntenium {

/// Placeholder for future C++ circuit execution from serialised IR.
struct CircuitExecResult {
    std::map<std::string, int> counts;
    int shots;
    int num_qubits;
};

/// Execute a circuit from its JSON-serialised IR representation.
/// v0.1: Not yet implemented — returns empty result.
inline CircuitExecResult execute_ir_json(
    const std::string& /*ir_json*/,
    int /*shots*/ = 1024)
{
    return CircuitExecResult{{}, 0, 0};
}

} // namespace zuntenium
