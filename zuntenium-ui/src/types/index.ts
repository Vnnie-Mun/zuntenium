// src/types/index.ts
// ── All types match Zuntenium backend schemas exactly ──────────────────────

/* ── Circuit / Gate ──────────────────────────────────────────────────────── */

export interface GateSchema {
  name: string;        // lowercase gate name: "h", "cx", "rx", etc.
  qubits: number[];    // 0-based qubit indices
  params: number[];    // rotation angles in radians
}

export interface CircuitSchema {
  num_qubits: number;
  gates: GateSchema[];
  name?: string;
}

/* ── Job submission (POST /jobs) ─────────────────────────────────────────── */

export type JobType       = "circuit" | "hybrid" | "variational";
export type BackendChoice = "auto" | "statevector" | "tensornetwork" | "stabilizer" | "gpu_statevector" | "ibm" | "ionq" | "braket" | "rigetti";
export type OptimMode     = "rule" | "ai";

export interface JobSubmitRequest {
  job_type:           JobType;
  circuit?:           CircuitSchema;
  hybrid_config?:     Record<string, unknown>;
  parameters?:        Record<string, number>;
  shots:              number;
  backend:            BackendChoice;
  mitigation_enabled: boolean;
  optimization_mode:  OptimMode;
  max_bond_dim?:      number;
  priority:           number;   // 1–10
}

/* ── Job response ─────────────────────────────────────────────────────────── */

export type JobStatus = "pending" | "queued" | "running" | "completed" | "failed" | "cancelled" | "retrying";

export interface JobResponse {
  id:           string;
  status:       JobStatus;
  job_type:     JobType;
  created_at:   string;
  queued_at?:   string;
  started_at?:  string;
  completed_at?: string;
  retry_count:  number;
  last_error?:  string;
  user_id:      string;
  queue_name:   string;
  priority:     number;
}

export interface JobListResponse {
  jobs:      JobResponse[];
  total:     number;
  page:      number;
  page_size: number;
  pages:     number;
}

/* ── Job result (GET /jobs/{id}/result) ────────────────────────────────────── */

export interface JobResultResponse {
  job_id:              string;
  status:              JobStatus;
  counts?:             Record<string, number>;
  statevector?:        [number, number][];   // [[real, imag], ...]
  expectation_values?: Record<string, number>;
  energy?:             number;
  backend_used?:       string;
  decision_reason?:    string;
  ai_confidence?:      number;
  mitigation_method?:  string;
  fidelity_before?:    number;
  fidelity_after?:     number;
  truncation_error?:   number;
  execution_time_s?:   number;
  queue_time_s?:       number;
  total_time_s?:       number;
}

/* ── Backend info (GET /backends) ────────────────────────────────────────── */

export interface BackendInfo {
  name:         string;
  available:    boolean;
  description:  string;
  max_qubits?:  number;
  queue_depth?: number;
  avg_wait_s?:  number;
}

/* ── WebSocket messages ──────────────────────────────────────────────────── */

export type WSMessageType =
  | "job_status_update"
  | "job_completed"
  | "job_failed"
  | "pipeline_step"
  | "heartbeat";

export interface WSMessage {
  type:    WSMessageType;
  job_id?: string;
  data?:   Record<string, unknown>;
}

/* ── UI-only circuit state ───────────────────────────────────────────────── */

export type GateName =
  | "H" | "X" | "Y" | "Z" | "S" | "T" | "SDG" | "TDG" | "ID" | "SX"
  | "RX" | "RY" | "RZ" | "P"
  | "CX" | "CZ" | "SWAP" | "CY" | "CP"
  | "CCX";

export interface UIGate {
  id:       string;
  name:     GateName;
  qubit:    number;       // primary (control for 2q gates)
  target?:  number;       // for 2-qubit gates
  target2?: number;       // for Toffoli
  col:      number;       // column/time-step on the wire
  params:   number[];     // radians
}

export interface UICircuit {
  name:      string;
  numQubits: number;
  gates:     UIGate[];
}

/* ── Validation ──────────────────────────────────────────────────────────── */

export type ValidationSeverity = "error" | "warning" | "info";

export interface ValidationIssue {
  severity:  ValidationSeverity;
  gateId?:   string;
  message:   string;
  field?:    string;
}

export interface ValidationResult {
  valid:  boolean;
  issues: ValidationIssue[];
}

/* ── Execution pipeline ──────────────────────────────────────────────────── */

export type PipelineStepId =
  | "circuit"
  | "transpiler"
  | "ai"
  | "selector"
  | "mhos"
  | "execution"
  | "mitigation";

export type PipelineStepStatus = "idle" | "active" | "complete" | "error";

export interface PipelineStep {
  id:      PipelineStepId;
  label:   string;
  detail:  string;
  status:  PipelineStepStatus;
  badge?:  string;
  badgeClass?: string;
}

/* ── Gate metadata (UI layer) ─────────────────────────────────────────────── */

export interface GateMeta {
  label:       string;
  fullName:    string;
  description: string;
  color:       string;
  bg:          string;
  category:    "single" | "parametric" | "two-qubit" | "multi-qubit";
  hasParams:   boolean;
  paramNames?: string[];
  paramDefaults?: number[];
  isTwoQubit?: boolean;
  isThreeQubit?: boolean;
  backendName: string;   // lowercase name for Circuit JSON
}
