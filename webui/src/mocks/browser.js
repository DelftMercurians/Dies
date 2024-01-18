/**
 * browser.js
 *
 * Contains the setup for the mock server
 */
// ---------------- IMPORTS ----------------
import { setupWorker } from "msw/browser";
import { handlers } from "./handlers";

export const worker = setupWorker(...handlers);
