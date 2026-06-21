import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import * as path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    host: "0.0.0.0",
    proxy: {
      // `ws: true` lets the WebSocket at /api/ws ride the same proxy, so in dev
      // the socket is same-origin with the page (localhost:5173) just like prod.
      "/api": { target: `http://127.0.0.1:5555`, ws: true },
    },
    headers: {
      "Content-Security-Policy":
        "script-src 'self' 'unsafe-eval' 'unsafe-inline'",
    },
  },
  clearScreen: false,
  optimizeDeps: {
    esbuildOptions: {
      target: "esnext",
    },
  },
  build: {
    target: "es2020",
    outDir: "../crates/dies-webui/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        inlineDynamicImports: false,
        format: "iife",
        manualChunks: () => {
          return "Any string";
        },
      },
    },
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
