import { defineConfig } from "vite";
import wasmPack from "vite-plugin-wasm-pack";

export default defineConfig({
  appType: "mpa",
  build: {
    minify: false,
  },
  plugins: [wasmPack("../crates/dies-simulator-wasm")],
});
