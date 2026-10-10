import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "Fira Code", "monospace"],
      },
      colors: {
        /* ── Base surfaces ── */
        base:    "#040C1A",
        surface: "#07101F",
        card:    "#0C1829",
        hover:   "#101F35",
        active:  "#14253F",

        /* ── Borders ── */
        border:  {
          DEFAULT: "#162440",
          hi:      "#1D3058",
          glow:    "#1A3A6A",
        },

        /* ── Brand ── */
        primary: {
          DEFAULT: "#0084FF",
          dim:     "rgba(0,132,255,0.15)",
          glow:    "rgba(0,132,255,0.35)",
          600:     "#006ACC",
        },
        cyan: {
          DEFAULT: "#00D4B8",
          dim:     "rgba(0,212,184,0.15)",
          glow:    "rgba(0,212,184,0.35)",
        },

        /* ── Semantic ── */
        success: "#2ECC71",
        warning: "#F5A623",
        danger:  "#FF4757",
        info:    "#3B82F6",
        purple:  "#A855F7",

        /* ── Text ── */
        t1: "#E8F0FF",
        t2: "#8DA4C8",
        t3: "#4A6080",
        t4: "#2A3F60",

        /* ── Gate colors ── */
        "gate-h":    "#0084FF",
        "gate-x":    "#FF4757",
        "gate-y":    "#FF6B35",
        "gate-z":    "#A855F7",
        "gate-s":    "#00D4B8",
        "gate-t":    "#F5A623",
        "gate-rx":   "#E91E8C",
        "gate-ry":   "#FF6B35",
        "gate-rz":   "#7C3AED",
        "gate-cx":   "#0084FF",
        "gate-cz":   "#00D4B8",
        "gate-swap": "#F59E0B",
        "gate-ccx":  "#10B981",
      },

      boxShadow: {
        card:    "0 4px 24px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.03)",
        glow:    "0 0 20px rgba(0,132,255,0.3)",
        "glow-cyan": "0 0 20px rgba(0,212,184,0.3)",
        "glow-sm":   "0 0 8px rgba(0,132,255,0.4)",
        gate:    "0 2px 8px rgba(0,0,0,0.5)",
        "gate-selected": "0 0 0 2px currentColor, 0 0 12px currentColor",
      },

      animation: {
        "pulse-slow":  "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "spin-slow":   "spin 2s linear infinite",
        "glow-pulse":  "glowPulse 2s ease-in-out infinite",
        "slide-in":    "slideIn 0.3s ease forwards",
        "fade-in":     "fadeIn 0.2s ease forwards",
        "fill-bar":    "fillBar 0.6s ease forwards",
      },

      keyframes: {
        glowPulse: {
          "0%,100%": { opacity: "1" },
          "50%": { opacity: "0.5" },
        },
        slideIn: {
          from: { transform: "translateX(16px)", opacity: "0" },
          to:   { transform: "translateX(0)",    opacity: "1" },
        },
        fadeIn: {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
        fillBar: {
          from: { width: "0%" },
          to:   { width: "var(--fill-w)" },
        },
      },

      backgroundImage: {
        "grid-dark": `
          linear-gradient(rgba(22,36,64,0.4) 1px, transparent 1px),
          linear-gradient(90deg, rgba(22,36,64,0.4) 1px, transparent 1px)
        `,
        "radial-glow": "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,132,255,0.08) 0%, transparent 60%)",
      },
      backgroundSize: {
        grid: "40px 40px",
      },
    },
  },
  plugins: [],
};

export default config;
