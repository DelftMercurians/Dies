import { defineConfig, ViteDevServer } from "vite";
import open from "open";
import { svelte } from "@sveltejs/vite-plugin-svelte";

/**
 * A vite plugin that only open the UI in a browser if it's not already open.
 */
const openBrowserPlugin = (enabled: boolean = true) => ({
  name: "open-browser",
  configureServer(server: ViteDevServer) {
    // return a post hook that is called after internal middlewares are
    // installed
    return () => {
      if (!enabled) return;

      // wait a bit then check if we have incoming websocket connections
      setTimeout(() => {
        if (server.ws.clients.size === 0) {
          // if not, open the browser
          server.config.logger.info("Opening browser...");
          const adr = server.httpServer.address();
          const url =
            typeof adr === "string" ? adr : `http://localhost:${adr.port}`;
          open(url);
        }
      }, 2000);
    };
  },
});

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    // host: process.env.HOST || undefined,
    // port: parseInt(process.env.PORT) || undefined,
    // strictPort: !!parseInt(process.env.PORT),
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
  plugins: [
    svelte(),
    // openBrowserPlugin(process.env.OPEN_BROWSER !== undefined),
  ],
});
