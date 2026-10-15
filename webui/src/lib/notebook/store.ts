/**
 * Notebook state in the browser. The server owns notebooks and cell results
 * (`/api/notebooks`); this is a cache of the open ones plus live run state fed
 * by `Notebook` WS events via {@link applyNotebookEvent}.
 */
import { atom, getDefaultStore } from "jotai";
import {
  CellRunState,
  LogNotebook,
  Notebook,
  NotebookCell,
  NotebookDoc,
  NotebookEvent,
  NotebookRobot,
  NotebookSummary,
} from "@/bindings";

export type Cell = NotebookCell;

/** What `show()` produced, as encoded by the engine prelude. */
export type ShowData =
  | { type: "plot"; spec: unknown }
  | { type: "table"; rows: Record<string, unknown>[] }
  | { type: "text"; text: string }
  | { type: "json"; value: unknown };

export type CellOutput =
  | { kind: "show"; data: ShowData }
  | { kind: "error"; text: string }
  | { kind: "aborted"; text: string };

export type CellStatus = "idle" | "queued" | "running" | "done" | "error" | "aborted";

export interface CellRun {
  status: CellStatus;
  outputs: CellOutput[];
  startedAt?: number;
  finishedAt?: number;
}

export interface NotebookMessage {
  ts: number;
  kind: "error" | "info";
  text: string;
}

export const newId = (prefix: string) =>
  `${prefix}${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;

/** Open notebooks by id (loaded from the server). */
export const notebooksAtom = atom<Record<string, Notebook>>({});
/** Run state per cell id (cell ids are unique across notebooks). */
export const cellRunsAtom = atom<Record<string, CellRun>>({});
/** Notebook-wide messages, keyed by notebook id ("" = engine-wide). */
export const notebookMessagesAtom = atom<Record<string, NotebookMessage[]>>({});

const MAX_MESSAGES = 50;
const store = () => getDefaultStore();

// ---------------------------------------------------------------- server API

async function http<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${method} ${url}: HTTP ${r.status}`);
  const text = await r.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const listNotebooks = () => http<NotebookSummary[]>("GET", "/api/notebooks");
export const deleteNotebook = (id: string) => http<void>("DELETE", `/api/notebooks/${id}`);

/** Notebooks stored in a recorded log. */
export const listLogNotebooks = (path: string) =>
  http<LogNotebook[]>("GET", `/api/log-notebooks?path=${encodeURIComponent(path)}`);
/** Copy a log's notebook into the notebook list (once) and return it. */
export const openLogNotebook = (path: string, id: string) =>
  http<NotebookSummary>("POST", "/api/log-notebooks/open", { path, id });

export async function createNotebook(title: string, codes: string[]): Promise<Notebook> {
  const id = newId("n");
  const nb: Notebook = {
    id,
    title,
    cells: (codes.length ? codes : [""]).map((code) => ({ id: newId("c"), code })),
    updated_ms: 0,
  };
  return http<Notebook>("PUT", `/api/notebooks/${id}`, nb);
}

function decodeRun(r: CellRunState): CellRun {
  const outputs: CellOutput[] = r.outputs.map((o) => {
    const data = parseData(o.data);
    if (o.kind === "show") return { kind: "show", data: data as ShowData };
    const text = typeof data === "string" ? data : JSON.stringify(data);
    return { kind: o.kind === "error" ? "error" : "aborted", text };
  });
  return {
    status: r.status as CellStatus,
    outputs,
    startedAt: r.started_ms,
    finishedAt: r.finished_ms,
  };
}

/** Load a notebook and its stored cell results into the cache. */
export async function loadNotebook(id: string): Promise<Notebook> {
  const doc = await http<NotebookDoc>("GET", `/api/notebooks/${id}`);
  const s = store();
  s.set(notebooksAtom, { ...s.get(notebooksAtom), [id]: doc.notebook });
  const runs = { ...s.get(cellRunsAtom) };
  for (const [cell, run] of Object.entries(doc.runs)) {
    // Live state (if any) is newer than the stored copy.
    if (!runs[cell]) runs[cell] = decodeRun(run);
  }
  s.set(cellRunsAtom, runs);
  return doc.notebook;
}

const saveTimers = new Map<string, ReturnType<typeof setTimeout>>();

/** Update a cached notebook and save it (debounced). */
export function updateNotebook(id: string, f: (nb: Notebook) => Notebook) {
  const s = store();
  const cur = s.get(notebooksAtom)[id];
  if (!cur) return;
  const next = f(cur);
  s.set(notebooksAtom, { ...s.get(notebooksAtom), [id]: next });
  clearTimeout(saveTimers.get(id));
  saveTimers.set(
    id,
    setTimeout(() => {
      saveTimers.delete(id);
      const latest = store().get(notebooksAtom)[id];
      if (latest) http("PUT", `/api/notebooks/${id}`, latest).catch((e) => console.error(e));
    }, 500)
  );
}

export const setNotebookRobot = (id: string, robot: NotebookRobot | undefined) =>
  updateNotebook(id, (nb) => ({ ...nb, robot }));

/** Move cells from the earlier localStorage-only notebook to the server, once. */
export async function migrateLocalNotebook() {
  const key = "dies-notebook-cells";
  const raw = localStorage.getItem(key);
  if (!raw) return;
  try {
    const cells = JSON.parse(raw) as { code: string }[];
    if (cells.some((c) => c.code.trim())) await createNotebook("Scratch", cells.map((c) => c.code));
    localStorage.removeItem(key);
  } catch (e) {
    console.error("Notebook migration failed", e);
  }
}

// ---------------------------------------------------------------- run state

function parseData(data: string | undefined | null): unknown {
  if (data == null) return undefined;
  try {
    return JSON.parse(data);
  } catch {
    return data;
  }
}

function pushMessage(notebook: string, kind: NotebookMessage["kind"], text: string) {
  const s = store();
  const all = s.get(notebookMessagesAtom);
  const cur = all[notebook] ?? [];
  s.set(notebookMessagesAtom, {
    ...all,
    [notebook]: [...cur.slice(-(MAX_MESSAGES - 1)), { ts: Date.now(), kind, text }],
  });
}

export function clearMessages(notebook: string) {
  const s = store();
  s.set(notebookMessagesAtom, { ...s.get(notebookMessagesAtom), [notebook]: [] });
}

function updateRun(cell: string, f: (run: CellRun) => CellRun) {
  const s = store();
  const runs = s.get(cellRunsAtom);
  const run = runs[cell] ?? { status: "idle", outputs: [] };
  s.set(cellRunsAtom, { ...runs, [cell]: f(run) });
}

/** Mark a cell as submitted: clear its outputs and wait for `started`. */
export function markQueued(cell: string) {
  updateRun(cell, () => ({ status: "queued", outputs: [], startedAt: Date.now() }));
}

/** Show a client-side failure (e.g. compile error) on a cell. */
export function markFailed(cell: string, text: string) {
  updateRun(cell, () => ({
    status: "error",
    outputs: [{ kind: "error", text }],
    finishedAt: Date.now(),
  }));
}

/** Mark active cells as aborted — all, or only those in `cellIds`. */
export function abortActiveCells(reason: string, cellIds?: Set<string>) {
  const s = store();
  const runs = s.get(cellRunsAtom);
  let changed = false;
  const next: Record<string, CellRun> = {};
  for (const [id, run] of Object.entries(runs)) {
    const active = run.status === "running" || run.status === "queued";
    if (active && (!cellIds || cellIds.has(id))) {
      changed = true;
      next[id] = {
        ...run,
        status: "aborted",
        outputs: [...run.outputs, { kind: "aborted", text: reason }],
        finishedAt: Date.now(),
      };
    } else {
      next[id] = run;
    }
  }
  if (changed) s.set(cellRunsAtom, next);
}

export function applyNotebookEvent(ev: NotebookEvent) {
  const data = parseData(ev.data);
  const text = typeof data === "string" ? data : JSON.stringify(data);

  if (ev.cell == null) {
    const nb = ev.notebook ?? "";
    if (ev.kind === "reset") {
      const cells = store().get(notebooksAtom)[nb]?.cells ?? [];
      abortActiveCells("notebook restarted", new Set(cells.map((c) => c.id)));
    } else if (ev.kind === "error") {
      pushMessage(nb, "error", text);
    } else {
      pushMessage(nb, "info", `${ev.kind}${data !== undefined ? `: ${text}` : ""}`);
    }
    return;
  }

  const cell = ev.cell;
  switch (ev.kind) {
    case "queued":
      updateRun(cell, (r) => (r.status === "queued" ? r : { ...r, status: "queued" }));
      break;
    case "started":
      updateRun(cell, (r) => ({ ...r, status: "running", startedAt: Date.now() }));
      break;
    case "show":
      updateRun(cell, (r) => ({ ...r, outputs: [...r.outputs, { kind: "show", data: data as ShowData }] }));
      break;
    case "done":
      updateRun(cell, (r) => ({ ...r, status: "done", finishedAt: Date.now() }));
      break;
    case "error":
      updateRun(cell, (r) => ({
        ...r,
        status: "error",
        outputs: [...r.outputs, { kind: "error", text }],
        finishedAt: Date.now(),
      }));
      break;
    case "aborted":
      updateRun(cell, (r) => ({
        ...r,
        status: "aborted",
        outputs: [...r.outputs, { kind: "aborted", text }],
        finishedAt: Date.now(),
      }));
      break;
  }
}
