import { defineConfig, ViteDevServer } from "vite";
import open from "open";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      "/api": `http://127.0.0.1:5555`,
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
  },
  plugins: [react()],
});
