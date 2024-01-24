/**
 * main.ts
 *
 * Bootstraps Vuetify and other plugins then mounts the App`
 */

// Plugins
import { registerPlugins } from "@/plugins";

// Components
import App from "./App.vue";

// Composables
import { createApp } from "vue";

const app = createApp(App);

async function enableMocking() {
  if (process.env.NODE_ENV !== "development") {
    return;
  }
  console.log("YO");

  const { worker } = await import("./mocks/browser.js");
  console.log("Yo");

  // `worker.start()` returns a Promise that resolves
  // once the Service Worker is up and ready to intercept requests.
  return worker.start();
}

registerPlugins(app);

enableMocking().then(() => {
  app.mount("#app");
});
