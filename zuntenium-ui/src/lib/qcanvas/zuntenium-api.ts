export interface CloudBackend {
  name: string;
  available: boolean;
  description: string;
  max_qubits?: number | null;
  queue_depth?: number | null;
  avg_wait_s?: number | null;
}

export interface CloudJob {
  id: string;
  status: string;
  last_error?: string | null;
}

export interface CloudJobResult {
  job_id: string;
  status: string;
  counts?: Record<string, number> | null;
  statevector?: [number, number][] | null;
  backend_used?: string | null;
  execution_time_s?: number | null;
  queue_time_s?: number | null;
  total_time_s?: number | null;
  fidelity_before?: number | null;
  fidelity_after?: number | null;
  mitigation_method?: string | null;
}

export interface SubmitCircuit {
  num_qubits: number;
  name: string;
  gates: { name: string; qubits: number[]; params: number[] }[];
}

const API_URL =
  process.env.NEXT_PUBLIC_API_URL !== undefined && process.env.NEXT_PUBLIC_API_URL !== ""
    ? process.env.NEXT_PUBLIC_API_URL
    : typeof window !== "undefined"
      ? ""
      : "http://127.0.0.1:8000";
const ACCESS_TOKEN_KEY = "quflow.access-token";
const REFRESH_TOKEN_KEY = "quflow.refresh-token";

export interface AuthSession {
  access_token: string;
  refresh_token: string;
}

export function saveSession(session: AuthSession): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, session.access_token);
  localStorage.setItem(REFRESH_TOKEN_KEY, session.refresh_token);
}

export function clearSession(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

export function storedAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

async function request<T>(
  path: string,
  token?: string,
  init: RequestInit = {},
  retryAuth = true,
): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  });
  if (response.status === 401 && token && retryAuth && typeof window !== "undefined") {
    const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
    if (refreshToken) {
      const refreshResponse = await fetch(
        `${API_URL}/auth/refresh?token=${encodeURIComponent(refreshToken)}`,
        { method: "POST" },
      );
      if (refreshResponse.ok) {
        const session = (await refreshResponse.json()) as AuthSession;
        saveSession(session);
        return request<T>(path, session.access_token, init, false);
      }
      clearSession();
    }
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string; error?: string } | null;
    throw new Error(body?.detail ?? body?.error ?? `QuFlow API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function registerAccount(email: string, username: string, password: string): Promise<void> {
  await request("/auth/register", undefined, {
    method: "POST",
    body: JSON.stringify({ email, username, password }),
  });
}

export function loginAccount(username: string, password: string): Promise<AuthSession> {
  return request("/auth/token", undefined, {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function fetchBackends(token: string): Promise<{ backends: CloudBackend[] }> {
  return request("/backends", token);
}

export function submitCircuit(token: string, circuit: SubmitCircuit, options: {
  backend: string;
  shots: number;
  mitigation: boolean;
}): Promise<CloudJob> {
  return request("/jobs", token, {
    method: "POST",
    body: JSON.stringify({
      job_type: "circuit",
      circuit,
      backend: options.backend,
      shots: options.shots,
      mitigation_enabled: options.mitigation,
      optimization_mode: "rule",
      priority: 5,
    }),
  });
}

export function fetchJob(token: string, jobId: string): Promise<CloudJob> {
  return request(`/jobs/${encodeURIComponent(jobId)}`, token);
}

export function fetchJobResult(token: string, jobId: string): Promise<CloudJobResult> {
  return request(`/jobs/${encodeURIComponent(jobId)}/result`, token);
}

export function cancelJob(token: string, jobId: string): Promise<{ message: string }> {
  return request(`/jobs/${encodeURIComponent(jobId)}/cancel`, token, { method: "POST" });
}