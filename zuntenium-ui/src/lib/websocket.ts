// src/lib/websocket.ts
// ── Robust WebSocket client with reconnection + heartbeat ──────────────────
import { useEffect, useRef, useCallback } from "react";
import type { WSMessage } from "@/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws";

interface WSOptions {
  onMessage:    (msg: WSMessage) => void;
  onConnect?:   () => void;
  onDisconnect?:() => void;
  jobId?:       string;
}

export function useWebSocket({ onMessage, onConnect, onDisconnect, jobId }: WSOptions) {
  const wsRef      = useRef<WebSocket | null>(null);
  const attemptsRef= useRef(0);
  const timerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    const url = jobId ? `${WS_URL}/jobs/${jobId}` : `${WS_URL}/platform`;
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) { ws.close(); return; }
        attemptsRef.current = 0;
        onConnect?.();
        // Heartbeat every 30s
        heartbeatRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "heartbeat" }));
          }
        }, 30_000);
      };

      ws.onmessage = (e) => {
        if (!mountedRef.current) return;
        try {
          const msg = JSON.parse(e.data as string) as WSMessage;
          onMessage(msg);
        } catch { /* ignore malformed messages */ }
      };

      ws.onerror = () => { /* handled by onclose */ };

      ws.onclose = () => {
        if (heartbeatRef.current) clearInterval(heartbeatRef.current);
        onDisconnect?.();
        if (!mountedRef.current) return;
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, cap at 30s
        const delay = Math.min(1000 * 2 ** attemptsRef.current, 30_000);
        attemptsRef.current++;
        timerRef.current = setTimeout(connect, delay);
      };
    } catch {
      const delay = Math.min(1000 * 2 ** attemptsRef.current, 30_000);
      attemptsRef.current++;
      timerRef.current = setTimeout(connect, delay);
    }
  }, [jobId, onMessage, onConnect, onDisconnect]);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (timerRef.current)     clearTimeout(timerRef.current);
      if (heartbeatRef.current) clearInterval(heartbeatRef.current);
      if (wsRef.current)        wsRef.current.close();
    };
  }, [connect]);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}

// ── API client ────────────────────────────────────────────────────────────
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function apiRequest<T>(
  path: string,
  options?: RequestInit,
  token?: string
): Promise<T> {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options?.headers ?? {}),
  };
  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText })) as { error: string };
    throw new Error(err.error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

import type {
  JobSubmitRequest, JobResponse, JobResultResponse, BackendInfo,
  JobListResponse,
} from "@/types";

export const api = {
  submitJob:   (body: JobSubmitRequest, token?: string) =>
    apiRequest<JobResponse>("/jobs", { method:"POST", body: JSON.stringify(body) }, token),

  getJob:      (id: string, token?: string) =>
    apiRequest<JobResponse>(`/jobs/${id}`, {}, token),

  getResult:   (id: string, token?: string) =>
    apiRequest<JobResultResponse>(`/jobs/${id}/result`, {}, token),

  listJobs:    (params?: { status?: string; page?: number }, token?: string) => {
    const qs = new URLSearchParams(params as Record<string,string>).toString();
    return apiRequest<JobListResponse>(`/jobs${qs ? `?${qs}` : ""}`, {}, token);
  },

  cancelJob:   (id: string, token?: string) =>
    apiRequest<{ message: string }>(`/jobs/${id}/cancel`, { method:"POST" }, token),

  listBackends:(token?: string) =>
    apiRequest<{ backends: BackendInfo[] }>("/backends", {}, token),

  adminStats:  (token?: string) =>
    apiRequest<Record<string,unknown>>("/admin/stats", {}, token),
};
