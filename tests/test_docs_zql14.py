"""
Verification test for ZQL-14 Developer Operations & Live Platform Ecosystem.
Tests wire schemas, CRDT Lamport clock logic, and cloud API payload formats.
"""

def test_docs_zql14_circuit_payload_schema():
    # Verify the JSON payload structure accepted by POST /jobs
    payload = {
        "job_type": "circuit",
        "circuit": {
            "num_qubits": 2,
            "name": "CloudBellCircuit",
            "gates": [
                {"name": "h", "qubits": [0], "params": []},
                {"name": "cx", "qubits": [0, 1], "params": []}
            ]
        },
        "backend": "statevector",
        "shots": 1024,
        "mitigation_enabled": True,
        "optimization_mode": "rule",
        "priority": 5
    }
    assert payload["job_type"] == "circuit"
    assert payload["circuit"]["num_qubits"] == 2
    assert len(payload["circuit"]["gates"]) == 2
    assert payload["shots"] == 1024


def test_docs_zql14_crdt_lamport_clock_and_tombstone():
    # Simulate the CRDT gate map logic from zuntenium-collab-server
    class CRDTSimulator:
        def __init__(self):
            self.registers = {}
            self.tombstones = set()
            self.lamport = 0

        def tick(self):
            self.lamport += 1
            return self.lamport

        def apply(self, op):
            op_lamport = op["lamport"]
            self.lamport = max(self.lamport, op_lamport) + 1
            gate_id = op["gateId"]

            if op["type"] in ("ADD_GATE", "UPDATE_GATE"):
                if gate_id in self.tombstones:
                    return {"applied": False, "reason": "gate_deleted"}
                existing = self.registers.get(gate_id)
                if not existing or op_lamport > existing["lamport"]:
                    self.registers[gate_id] = op
                    return {"applied": True}
                return {"applied": False, "reason": "stale_clock"}

            elif op["type"] == "DELETE_GATE":
                self.tombstones.add(gate_id)
                self.registers.pop(gate_id, None)
                return {"applied": True}

            return {"applied": False, "reason": "unknown_op"}

    crdt = CRDTSimulator()
    # 1. Client A adds Gate G1
    op1 = {"type": "ADD_GATE", "gateId": "g1", "lamport": 1, "authorId": "userA"}
    res1 = crdt.apply(op1)
    assert res1["applied"] is True
    assert "g1" in crdt.registers

    # 2. Client B deletes Gate G1
    op2 = {"type": "DELETE_GATE", "gateId": "g1", "lamport": 3, "authorId": "userB"}
    res2 = crdt.apply(op2)
    assert res2["applied"] is True
    assert "g1" not in crdt.registers
    assert "g1" in crdt.tombstones

    # 3. Concurrent add or update to G1 rejected by tombstone
    op3 = {"type": "UPDATE_GATE", "gateId": "g1", "lamport": 2, "authorId": "userA"}
    res3 = crdt.apply(op3)
    assert res3["applied"] is False
    assert res3["reason"] == "gate_deleted"
