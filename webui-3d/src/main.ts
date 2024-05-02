import "./style.css";

import init, { greet } from "dies-simulator-wasm";

init().then(() => {
  console.log("init wasm-pack");
  greet();
});
