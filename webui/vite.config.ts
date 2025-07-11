import { defineConfig, ViteDevServer } from "vite";
import react from "@vitejs/plugin-react";
import * as path from "path";

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      "/api": `http://127.0.0.1:5555`,
    },
    headers: {
      "Content-Security-Policy":
        "script-src 'self' 'unsafe-eval' 'unsafe-inline'",
    },
  },
  // logLevel: "error",
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
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
