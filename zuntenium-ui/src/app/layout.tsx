// src/app/layout.tsx
import type { Metadata, Viewport } from "next";
import "./globals.css";
import "./qcanvas.css";
import { Providers } from "@/components/layout/Providers";

export const metadata: Metadata = {
  title: "QuFlow — Quantum Workspace",
  description: "QuFlow Quantum Computing IDE — Circuit Builder, AI Optimizer, MHOS, Error Mitigation",
};

export const viewport: Viewport = {
  themeColor: "#040C1A",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-base text-t1 font-sans antialiased overflow-hidden h-screen flex flex-col">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
