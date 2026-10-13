/**
 * One-time Monaco setup for notebook cells: TS language service workers, and
 * the notebook API + world types as extra libs so cells get autocompletion and
 * type errors. Cell models are TS *scripts* (no imports/exports), so all cells
 * share one global scope — mirroring how the engine runs them.
 */
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";
import bindingsSource from "@/bindings.ts?raw";

let setupDone: Promise<void> | null = null;

// Diagnostics that are wrong for notebook semantics: cross-cell redeclaration
// (re-running/redefining across cells is the point) and top-level await/return
// (cells run inside an async wrapper).
const IGNORED_DIAGNOSTICS = [2451, 2300, 2393, 1375, 1378, 1108, 1308];

export function ensureMonacoWorkers() {
  const g = self as any;
  if (!g.MonacoEnvironment) {
    g.MonacoEnvironment = {
      getWorker(_: string, label: string) {
        if (label === "typescript" || label === "javascript") return new tsWorker();
        return new editorWorker();
      },
    };
  }
}

/** Editor theme matching the app palette (index.css tokens). */
export const NOTEBOOK_THEME = "dies-notebook";

export function setupNotebookMonaco(): Promise<void> {
  if (setupDone) return setupDone;
  ensureMonacoWorkers();
  monaco.editor.defineTheme(NOTEBOOK_THEME, {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: {
      "editor.background": "#12141a", // bg-surface
      "editorGutter.background": "#12141a",
      "editorLineNumber.foreground": "#5a6270", // text-muted
      "editorLineNumber.activeForeground": "#a0a8b4", // text-std
      "editorWidget.background": "#1a1d24", // bg-elevated
      "editorWidget.border": "#3a4250", // border-std
      "editorSuggestWidget.background": "#1a1d24",
      "editorSuggestWidget.border": "#3a4250",
      "editorHoverWidget.background": "#1a1d24",
      "editorHoverWidget.border": "#3a4250",
    },
  });
  const ts = monaco.languages.typescript;
  ts.typescriptDefaults.setCompilerOptions({
    target: ts.ScriptTarget.ESNext,
    lib: ["esnext"],
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.NodeJs,
    allowNonTsExtensions: true,
    strict: false,
    noImplicitAny: false,
    noEmit: true,
  });
  ts.typescriptDefaults.setDiagnosticsOptions({
    noSemanticValidation: false,
    noSyntaxValidation: false,
    diagnosticCodesToIgnore: IGNORED_DIAGNOSTICS,
  });
  ts.typescriptDefaults.addExtraLib(bindingsSource, "file:///notebook/bindings.ts");
  setupDone = fetch("/api/notebook/types")
    .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((types) => {
      ts.typescriptDefaults.addExtraLib(types, "file:///notebook/prelude.d.ts");
    })
    .catch((e) => {
      console.warn("Notebook: failed to load API types", e);
    });
  return setupDone;
}

export const cellUri = (cellId: string) => monaco.Uri.parse(`file:///notebook/${cellId}.ts`);
