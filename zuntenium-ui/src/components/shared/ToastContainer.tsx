"use client";

import { X } from "lucide-react";
import { useEffect } from "react";
import { useUIStore } from "@/store";

const toastStyles = {
  success: "border-emerald-500/50 text-emerald-300",
  error: "border-rose-500/50 text-rose-300",
  warning: "border-amber-500/50 text-amber-300",
  info: "border-cyan-500/50 text-cyan-300",
} as const;

export function ToastContainer() {
  const toasts = useUIStore((state) => state.toasts);
  const removeToast = useUIStore((state) => state.removeToast);

  useEffect(() => {
    const timers = toasts.map((toast) =>
      window.setTimeout(() => removeToast(toast.id), 5000),
    );
    return () => timers.forEach(window.clearTimeout);
  }, [toasts, removeToast]);

  return (
    <div
      aria-label="Notifications"
      className="fixed bottom-5 right-5 z-50 flex w-[min(24rem,calc(100vw-2.5rem))] flex-col gap-2"
      role="region"
    >
      {toasts.map((toast) => (
        <section
          aria-live={toast.type === "error" ? "assertive" : "polite"}
          className={`border-l-2 bg-slate-950/95 p-3 shadow-xl ${toastStyles[toast.type]}`}
          key={toast.id}
          role={toast.type === "error" ? "alert" : "status"}
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-slate-100">{toast.title}</h2>
              <p className="mt-1 break-words text-xs text-slate-300">{toast.message}</p>
            </div>
            <button
              aria-label="Dismiss notification"
              className="shrink-0 text-slate-400 hover:text-white"
              onClick={() => removeToast(toast.id)}
              type="button"
            >
              <X aria-hidden="true" size={16} />
            </button>
          </div>
        </section>
      ))}
    </div>
  );
}