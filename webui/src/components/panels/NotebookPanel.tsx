/**
 * Notebook panels — interactive motion testing.
 *
 * One Dockview panel per notebook (params `{ notebookId }`); the same component
 * without params is the NOTEBOOKS launcher (list, new, tutorials). Notebooks
 * and cell results live on the server; cells are TypeScript, compiled in the
 * browser and run by the executor's embedded JS engine (crates/dies-notebook),
 * one scope per notebook, one cell at a time.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { DockviewApi, IDockviewPanelProps } from "dockview";
import * as monaco from "monaco-editor";
import { useAtomValue } from "jotai";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Download,
  FastForward,
  FilePlus,
  History,
  Play,
  Plus,
  RotateCcw,
  Square,
  Trash2,
  X,
} from "lucide-react";

import { replayStateAtom, sendWsCommand, usePrimaryTeam, useStatus, useWorldState } from "@/api";
import { Notebook, NotebookCmd, NotebookRobot, TeamColor } from "@/bindings";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Cell,
  CellRun,
  abortActiveCells,
  cellRunsAtom,
  clearMessages,
  createNotebook,
  deleteNotebook,
  listLogNotebooks,
  listNotebooks,
  loadNotebook,
  markFailed,
  markQueued,
  migrateLocalNotebook,
  newId,
  notebookMessagesAtom,
  notebooksAtom,
  openLogNotebook,
  setNotebookRobot,
  updateNotebook,
} from "@/lib/notebook/store";
import { compileCell } from "@/lib/notebook/transform";
import { NOTEBOOK_THEME, cellUri, setupNotebookMonaco } from "@/lib/notebook/monaco";
import { OutputView } from "@/lib/notebook/outputs";
import { NOTEBOOK_EXAMPLES } from "@/lib/notebook/examples";

const MAX_EDITOR_HEIGHT = 520;
/** `robot()` with no arguments — gets the inline robot chip. */
const ROBOT_SHORTHAND = /\brobot\(\s*\)/g;
/** Any `robot(...)` call — more than one per cell is linted. */
const ROBOT_CALL = /\brobot\s*\(/g;

function sendNotebook(cmd: NotebookCmd) {
  sendWsCommand({ type: "Notebook", data: cmd });
}

function stopAll() {
  try {
    sendNotebook({ type: "Abort" });
  } catch {
    toast.error("Not connected");
  }
}

const isInEditor = (target: EventTarget | null) =>
  target instanceof Element && !!target.closest(".monaco-editor");

export const notebookPanelId = (notebookId: string) => `notebook-${notebookId}`;

/** Open (or focus) a notebook's panel, as a tab next to `reference`. */
export function openNotebookPanel(api: DockviewApi, nb: { id: string; title: string }, reference?: string) {
  const id = notebookPanelId(nb.id);
  const existing = api.getPanel(id);
  if (existing) {
    existing.api.setActive();
    return;
  }
  const ref =
    (reference && api.getPanel(reference) ? reference : undefined) ??
    api.panels.find((p) => p.id.startsWith("notebook"))?.id;
  api.addPanel({
    id,
    component: "notebook",
    title: nb.title,
    params: { notebookId: nb.id },
    position: ref ? { referencePanel: ref, direction: "within" } : undefined,
  });
}

// ----------------------------------------------------------------------------
// Robots on the field
// ----------------------------------------------------------------------------

/** Our robots vision sees (primary team). */
function useVisibleRobots(): NotebookRobot[] {
  const world = useWorldState();
  const [team] = usePrimaryTeam();
  return useMemo(() => {
    if (world.status !== "connected") return [];
    const roster = team === TeamColor.Blue ? world.data.blue_team : world.data.yellow_team;
    return roster.map((p) => ({ team, id: p.id })).sort((a, b) => a.id - b.id);
  }, [world, team]);
}

interface RobotChoice {
  /** What `robot()` resolves to, or null if the user has to pick. */
  robot: NotebookRobot | null;
  label: string;
  warn: boolean;
}

function resolveRobot(nb: Notebook | undefined, visible: NotebookRobot[]): RobotChoice {
  const picked = nb?.robot;
  if (picked) {
    const seen = visible.some((r) => r.id === picked.id && r.team === picked.team);
    return { robot: picked, label: `#${picked.id}${seen ? "" : " (not seen)"}`, warn: !seen };
  }
  if (visible.length === 1) return { robot: visible[0], label: `#${visible[0].id} auto`, warn: false };
  return { robot: null, label: visible.length ? "pick robot" : "no robot", warn: true };
}

// ----------------------------------------------------------------------------
// Cell editor
// ----------------------------------------------------------------------------

interface CellEditorProps {
  notebookId: string;
  cell: Cell;
  focusToken: number;
  chip: RobotChoice;
  onChange: (code: string) => void;
  onRun: () => void;
  onRunAndAdvance: () => void;
  onChipClick: (anchor: DOMRect) => void;
}

const CellEditor: React.FC<CellEditorProps> = ({
  notebookId,
  cell,
  focusToken,
  chip,
  onChange,
  onRun,
  onRunAndAdvance,
  onChipClick,
}) => {
  const hostRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const decoRef = useRef<monaco.editor.IEditorDecorationsCollection | null>(null);
  const handlers = useRef({ onChange, onRun, onRunAndAdvance, onChipClick });
  handlers.current = { onChange, onRun, onRunAndAdvance, onChipClick };
  const chipRef = useRef(chip);
  chipRef.current = chip;

  // Robot chips inside `robot()` and the one-robot-per-cell lint.
  const annotate = useCallback(() => {
    const editor = editorRef.current;
    const model = editor?.getModel();
    if (!editor || !model) return;
    const text = model.getValue();
    const c = chipRef.current;
    const decorations: monaco.editor.IModelDeltaDecoration[] = [];
    for (const m of text.matchAll(ROBOT_SHORTHAND)) {
      const pos = model.getPositionAt(m.index! + "robot(".length);
      decorations.push({
        range: new monaco.Range(pos.lineNumber, pos.column, pos.lineNumber, pos.column),
        options: {
          after: { content: ` ${c.label} ▾ `, inlineClassName: c.warn ? "nb-chip nb-chip-warn" : "nb-chip" },
          showIfCollapsed: true,
        },
      });
    }
    decoRef.current?.set(decorations);

    const markers: monaco.editor.IMarkerData[] = [];
    [...text.matchAll(ROBOT_CALL)].slice(1).forEach((m) => {
      const start = model.getPositionAt(m.index!);
      const end = model.getPositionAt(m.index! + m[0].length);
      markers.push({
        severity: monaco.MarkerSeverity.Warning,
        message: "One robot per cell. Make a handle once: const R = robot()",
        startLineNumber: start.lineNumber,
        startColumn: start.column,
        endLineNumber: end.lineNumber,
        endColumn: end.column,
      });
    });
    monaco.editor.setModelMarkers(model, "notebook-lint", markers);
  }, []);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    setupNotebookMonaco();
    const uri = cellUri(`${notebookId}/${cell.id}`);
    const model =
      monaco.editor.getModel(uri) ?? monaco.editor.createModel(cell.code, "typescript", uri);
    if (model.getValue() !== cell.code) model.setValue(cell.code);

    const editor = monaco.editor.create(host, {
      model,
      theme: NOTEBOOK_THEME,
      minimap: { enabled: false },
      lineNumbers: "on",
      lineNumbersMinChars: 2,
      glyphMargin: false,
      folding: false,
      fontFamily: "JetBrains Mono, monospace",
      fontSize: 12,
      scrollBeyondLastLine: false,
      renderLineHighlight: "none",
      overviewRulerLanes: 0,
      hideCursorInOverviewRuler: true,
      scrollbar: { alwaysConsumeMouseWheel: false, verticalScrollbarSize: 8 },
      padding: { top: 4, bottom: 4 },
      fixedOverflowWidgets: true,
      tabSize: 2,
    });
    editorRef.current = editor;
    decoRef.current = editor.createDecorationsCollection();

    const fit = () => {
      const h = Math.min(MAX_EDITOR_HEIGHT, Math.max(22, editor.getContentHeight()));
      host.style.height = `${h}px`;
      editor.layout({ width: host.clientWidth, height: h });
    };
    const sizeSub = editor.onDidContentSizeChange(fit);
    const changeSub = editor.onDidChangeModelContent(() => {
      handlers.current.onChange(editor.getValue());
      annotate();
    });
    const mouseSub = editor.onMouseDown((e) => {
      // The chip is injected text styled with `nb-chip`; detect it by the DOM node.
      const el = e.event.browserEvent.target;
      const chipEl = el instanceof Element ? el.closest(".nb-chip") : null;
      if (chipEl) {
        handlers.current.onChipClick(chipEl.getBoundingClientRect());
      }
    });
    const ro = new ResizeObserver(fit);
    ro.observe(host);
    fit();
    annotate();

    // CtrlCmd is Cmd on macOS; also take real Ctrl+Enter there.
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => handlers.current.onRun());
    editor.addCommand(monaco.KeyMod.WinCtrl | monaco.KeyCode.Enter, () => handlers.current.onRun());
    editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Enter, () => handlers.current.onRunAndAdvance());
    editor.addCommand(monaco.KeyMod.Shift | monaco.KeyCode.Escape, () => stopAll());

    return () => {
      sizeSub.dispose();
      changeSub.dispose();
      mouseSub.dispose();
      ro.disconnect();
      editor.dispose();
      editorRef.current = null;
      // The model outlives the editor (view switches remount panels); it is
      // disposed when the cell is deleted.
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebookId, cell.id]);

  useEffect(() => {
    annotate();
  }, [chip.label, chip.warn, annotate]);

  useEffect(() => {
    if (focusToken > 0) editorRef.current?.focus();
  }, [focusToken]);

  return <div ref={hostRef} className="w-full" />;
};

// ----------------------------------------------------------------------------
// Cell
// ----------------------------------------------------------------------------

const STATUS_STYLE: Record<CellRun["status"], string> = {
  idle: "bg-border-muted",
  queued: "bg-text-dim animate-pulse",
  running: "bg-accent-cyan animate-pulse",
  done: "bg-accent-green",
  error: "bg-accent-red",
  aborted: "bg-accent-amber",
};

interface CellViewProps extends Omit<CellEditorProps, "onChange"> {
  index: number;
  count: number;
  run: CellRun | undefined;
  onChange: (code: string) => void;
  onDelete: () => void;
  onMove: (delta: number) => void;
  onAddBelow: () => void;
}

const CellView: React.FC<CellViewProps> = (props) => {
  const { index, count, run, onRun, onDelete, onMove, onAddBelow } = props;
  const status = run?.status ?? "idle";
  const duration =
    run?.startedAt && run.finishedAt ? (run.finishedAt - run.startedAt) / 1000 : null;

  return (
    <div className="nb-cell group border border-border-subtle bg-bg-surface">
      <div className="flex items-center gap-1 px-1 py-0.5 border-b border-border-subtle bg-bg-elevated/50">
        <SimpleTooltip title="Run (Ctrl/Cmd+Enter)">
          <Button variant="ghost" size="icon-xs" onClick={onRun} data-role="run">
            <Play className="h-3 w-3" />
          </Button>
        </SimpleTooltip>
        <span className={cn("h-2 w-2 rounded-full", STATUS_STYLE[status])} />
        <span className="text-xs text-text-dim tabular-nums">[{index + 1}]</span>
        <span className="text-xs text-text-dim" data-role="status">
          {status !== "idle" ? status : ""}
          {duration !== null && status !== "running" && status !== "queued"
            ? ` · ${duration.toFixed(1)}s`
            : ""}
        </span>
        <div className="ml-auto flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button variant="ghost" size="icon-xs" disabled={index === 0} onClick={() => onMove(-1)}>
            <ArrowUp className="h-3 w-3" />
          </Button>
          <Button variant="ghost" size="icon-xs" disabled={index === count - 1} onClick={() => onMove(1)}>
            <ArrowDown className="h-3 w-3" />
          </Button>
          <SimpleTooltip title="Add cell below">
            <Button variant="ghost" size="icon-xs" onClick={onAddBelow}>
              <Plus className="h-3 w-3" />
            </Button>
          </SimpleTooltip>
          <SimpleTooltip title="Delete cell">
            <Button variant="ghost" size="icon-xs" onClick={onDelete} className="hover:text-accent-red">
              <Trash2 className="h-3 w-3" />
            </Button>
          </SimpleTooltip>
        </div>
      </div>
      <CellEditor {...props} />
      {run && run.outputs.length > 0 && (
        <div className="flex flex-col gap-2 px-2 py-2 border-t border-border-subtle bg-bg-base/60">
          {run.outputs.map((o, i) => (
            <OutputView key={i} output={o} />
          ))}
        </div>
      )}
    </div>
  );
};

// ----------------------------------------------------------------------------
// Robot picker popup
// ----------------------------------------------------------------------------

const RobotPicker: React.FC<{
  at: DOMRect;
  robots: NotebookRobot[];
  current: NotebookRobot | undefined;
  onPick: (r: NotebookRobot | undefined) => void;
  onClose: () => void;
}> = ({ at, robots, current, onPick, onClose }) => {
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!(e.target instanceof Element && e.target.closest(".nb-robot-picker"))) onClose();
    };
    // Register after the click that opened us has finished propagating.
    const t = setTimeout(() => window.addEventListener("mousedown", close), 0);
    return () => {
      clearTimeout(t);
      window.removeEventListener("mousedown", close);
    };
  }, [onClose]);
  // Portal to <body>: Dockview panels are transformed, which would make a
  // `fixed` popup relative to the panel instead of the viewport.
  return createPortal(
    <div
      className="nb-robot-picker fixed z-50 min-w-[120px] border border-border-std bg-bg-overlay shadow-lg py-1 text-xs"
      style={{ left: at.left, top: at.bottom + 2 }}
    >
      <div className="px-2 py-0.5 text-text-dim">robot() is…</div>
      {robots.map((r) => (
        <button
          key={`${r.team}${r.id}`}
          className={cn(
            "block w-full text-left px-2 py-0.5 hover:bg-bg-elevated",
            current?.id === r.id && current?.team === r.team && "text-accent-cyan"
          )}
          onClick={() => onPick(r)}
        >
          #{r.id} <span className="text-text-dim">{r.team}</span>
        </button>
      ))}
      {robots.length === 0 && <div className="px-2 py-0.5 text-text-dim">no robots seen</div>}
      <button className="block w-full text-left px-2 py-0.5 hover:bg-bg-elevated text-text-dim" onClick={() => onPick(undefined)}>
        auto (only robot)
      </button>
    </div>,
    document.body
  );
};

// ----------------------------------------------------------------------------
// Notebook editor
// ----------------------------------------------------------------------------

const NotebookEditor: React.FC<{ notebookId: string; panel: IDockviewPanelProps }> = ({ notebookId, panel }) => {
  const notebooks = useAtomValue(notebooksAtom);
  const nb = notebooks[notebookId];
  const runs = useAtomValue(cellRunsAtom);
  const messages = useAtomValue(notebookMessagesAtom)[notebookId] ?? [];
  const { data: status } = useStatus();
  const executorRunning = status?.executor?.type === "RunningExecutor";
  const [focus, setFocus] = useState<{ id: string; token: number } | null>(null);
  const [picker, setPicker] = useState<DOMRect | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const visible = useVisibleRobots();
  const chip = resolveRobot(nb, visible);

  useEffect(() => {
    if (!nb) loadNotebook(notebookId).catch((e) => setLoadError(String(e?.message ?? e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebookId]);

  // Keep the tab title in sync with the notebook title.
  useEffect(() => {
    if (nb && panel.api.title !== nb.title) panel.api.setTitle(nb.title);
  }, [nb?.title, panel.api, nb]);

  const cells = nb?.cells ?? [];
  const cellIds = useMemo(() => new Set(cells.map((c) => c.id)), [cells]);
  const activeCount = cells.filter((c) => {
    const s = runs[c.id]?.status;
    return s === "running" || s === "queued";
  }).length;

  useEffect(() => {
    if (!executorRunning) abortActiveCells("executor stopped", cellIds);
  }, [executorRunning, cellIds]);

  // Esc outside the editor stops everything while something runs.
  useEffect(() => {
    if (activeCount === 0) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isInEditor(e.target)) stopAll();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeCount]);

  const update = (f: (nb: Notebook) => Notebook) => updateNotebook(notebookId, f);

  const send = (cell: Cell): boolean => {
    let source: string;
    try {
      source = compileCell(cell.id, cell.code);
    } catch (e: any) {
      markFailed(cell.id, String(e?.message ?? e));
      return false;
    }
    try {
      markQueued(cell.id);
      sendNotebook({
        type: "Eval",
        data: { notebook: notebookId, cell: cell.id, source, original: cell.code, robot: nb?.robot },
      });
      return true;
    } catch (e: any) {
      markFailed(cell.id, `Not sent: ${e?.message ?? e}`);
      return false;
    }
  };

  const checkReady = (codes: string[]): boolean => {
    if (!executorRunning) {
      toast.error("Start the executor first");
      return false;
    }
    if (codes.some((c) => c.match(ROBOT_SHORTHAND)) && !chip.robot) {
      toast.error(visible.length ? "Pick a robot: click the chip in robot()" : "No robot on the field");
      return false;
    }
    return true;
  };

  const runCell = (cell: Cell) => {
    if (checkReady([cell.code])) send(cell);
  };

  const runAll = () => {
    if (!checkReady(cells.map((c) => c.code))) return;
    try {
      sendNotebook({ type: "Reset", data: { notebook: notebookId } });
    } catch {
      toast.error("Not connected");
      return;
    }
    for (const c of cells) if (!send(c)) break;
  };

  const restart = () => {
    try {
      sendNotebook({ type: "Reset", data: { notebook: notebookId } });
    } catch {
      toast.error("Not connected");
    }
  };

  const addCell = (afterIndex: number, code = "") => {
    const cell = { id: newId("c"), code };
    update((n) => ({ ...n, cells: [...n.cells.slice(0, afterIndex + 1), cell, ...n.cells.slice(afterIndex + 1)] }));
    setFocus({ id: cell.id, token: Date.now() });
  };

  const deleteCell = (id: string) => {
    update((n) => ({ ...n, cells: n.cells.filter((c) => c.id !== id) }));
    monaco.editor.getModel(cellUri(`${notebookId}/${id}`))?.dispose();
  };

  const moveCell = (index: number, delta: number) =>
    update((n) => {
      const j = index + delta;
      if (j < 0 || j >= n.cells.length) return n;
      const next = [...n.cells];
      [next[index], next[j]] = [next[j], next[index]];
      return { ...n, cells: next };
    });

  const runAndAdvance = (index: number) => {
    runCell(cells[index]);
    if (index + 1 < cells.length) setFocus({ id: cells[index + 1].id, token: Date.now() });
    else addCell(index);
  };

  const exportNotebook = () => {
    if (!nb) return;
    const blob = new Blob([JSON.stringify({ title: nb.title, cells: nb.cells.map((c) => ({ code: c.code })) }, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${nb.title.replace(/[^\w-]+/g, "_") || "notebook"}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  if (!nb) {
    return (
      <div className="w-full h-full bg-bg-surface p-3 text-xs text-text-dim">
        {loadError ? <span className="text-accent-red">Could not load notebook: {loadError}</span> : "Loading…"}
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-bg-surface flex flex-col">
      {/* Toolbar */}
      <div className="flex items-center gap-1 px-2 py-1 border-b border-border-subtle bg-bg-elevated">
        <input
          className="bg-transparent text-xs text-text-std border border-transparent hover:border-border-muted focus:border-border-std px-1 py-0.5 w-44 outline-none"
          value={nb.title}
          onChange={(e) => update((n) => ({ ...n, title: e.target.value }))}
        />
        <SimpleTooltip title="Add cell">
          <Button variant="ghost" size="icon-xs" onClick={() => addCell(cells.length - 1)}>
            <Plus className="h-3 w-3" />
          </Button>
        </SimpleTooltip>
        <SimpleTooltip title="Export (.json)">
          <Button variant="ghost" size="icon-xs" onClick={exportNotebook}>
            <Download className="h-3 w-3" />
          </Button>
        </SimpleTooltip>
        <SimpleTooltip title="Robot used by robot()">
          <button
            className={cn("nb-chip ml-1 text-xs", chip.warn && "nb-chip-warn")}
            onClick={(e) => setPicker(e.currentTarget.getBoundingClientRect())}
          >
            robot(): {chip.label} ▾
          </button>
        </SimpleTooltip>

        <div className="ml-auto flex items-center gap-1">
          {!executorRunning && <span className="text-xs text-accent-amber mr-1">executor stopped</span>}
          {activeCount > 0 && (
            <span className="text-xs text-accent-cyan mr-1 tabular-nums">{activeCount} running</span>
          )}
          <SimpleTooltip title="Clear this notebook's variables">
            <Button variant="ghost" size="xs" onClick={restart} disabled={!executorRunning}>
              <RotateCcw className="h-3 w-3" /> Restart
            </Button>
          </SimpleTooltip>
          <SimpleTooltip title="Restart, then run every cell in order">
            <Button variant="ghost" size="xs" onClick={runAll} disabled={!executorRunning}>
              <FastForward className="h-3 w-3" /> Run all
            </Button>
          </SimpleTooltip>
          <SimpleTooltip title="Stop all notebooks, release robots (Esc / Shift+Esc in editor)">
            <Button variant="destructive" size="xs" onClick={stopAll} disabled={!executorRunning}>
              <Square className="h-3 w-3" /> Stop
            </Button>
          </SimpleTooltip>
        </div>
      </div>

      {messages.length > 0 && (
        <div className="flex items-start gap-2 px-2 py-1 border-b border-border-subtle bg-bg-base max-h-28 overflow-auto">
          <div className="flex-1 flex flex-col gap-0.5">
            {messages.slice(-4).map((m) => (
              <pre
                key={m.ts + m.text}
                className={cn("text-xs whitespace-pre-wrap", m.kind === "error" ? "text-accent-red" : "text-text-dim")}
              >
                {m.text}
              </pre>
            ))}
          </div>
          <Button variant="ghost" size="icon-xs" onClick={() => clearMessages(notebookId)}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      )}

      {/* Cells */}
      <div className="flex-1 overflow-auto px-2 py-2 flex flex-col gap-2">
        {cells.map((cell, i) => (
          <CellView
            key={cell.id}
            notebookId={notebookId}
            cell={cell}
            index={i}
            count={cells.length}
            run={runs[cell.id]}
            chip={chip}
            focusToken={focus?.id === cell.id ? focus.token : 0}
            onChange={(code) =>
              update((n) => ({ ...n, cells: n.cells.map((c) => (c.id === cell.id ? { ...c, code } : c)) }))
            }
            onRun={() => runCell(cell)}
            onRunAndAdvance={() => runAndAdvance(i)}
            onChipClick={(rect) => setPicker(rect)}
            onDelete={() => deleteCell(cell.id)}
            onMove={(d) => moveCell(i, d)}
            onAddBelow={() => addCell(i)}
          />
        ))}
        <button
          className="text-xs text-text-dim hover:text-text-std border border-dashed border-border-subtle py-1"
          onClick={() => addCell(cells.length - 1)}
        >
          + add cell
        </button>
      </div>

      {picker && (
        <RobotPicker
          at={picker}
          robots={visible}
          current={nb.robot}
          onPick={(r) => {
            setNotebookRobot(notebookId, r);
            setPicker(null);
          }}
          onClose={() => setPicker(null)}
        />
      )}
    </div>
  );
};

// ----------------------------------------------------------------------------
// Launcher
// ----------------------------------------------------------------------------

const NotebookLauncher: React.FC<{ panel: IDockviewPanelProps }> = ({ panel }) => {
  const queryClient = useQueryClient();
  const { data: notebooks, refetch } = useQuery({
    queryKey: ["notebooks"],
    queryFn: async () => {
      await migrateLocalNotebook();
      return listNotebooks();
    },
    refetchInterval: 5000,
  });

  const open = (nb: { id: string; title: string }) => openNotebookPanel(panel.containerApi, nb, panel.api.id);

  const replay = useAtomValue(replayStateAtom);
  const logPath = replay?.loaded ? replay.log_path : undefined;
  const { data: logNotebooks } = useQuery({
    queryKey: ["log-notebooks", logPath],
    queryFn: () => listLogNotebooks(logPath!),
    enabled: !!logPath,
  });

  const openFromLog = async (id: string) => {
    try {
      const nb = await openLogNotebook(logPath!, id);
      await queryClient.invalidateQueries({ queryKey: ["notebooks"] });
      open(nb);
    } catch (e: any) {
      toast.error(`Could not open notebook: ${e?.message ?? e}`);
    }
  };

  const create = async (title: string, codes: string[]) => {
    try {
      const nb = await createNotebook(title, codes);
      await queryClient.invalidateQueries({ queryKey: ["notebooks"] });
      open(nb);
    } catch (e: any) {
      toast.error(`Could not create notebook: ${e?.message ?? e}`);
    }
  };

  const remove = async (id: string, title: string) => {
    if (!window.confirm(`Delete notebook "${title}"?`)) return;
    try {
      await deleteNotebook(id);
      panel.containerApi.getPanel(notebookPanelId(id))?.api.close();
      refetch();
    } catch (e: any) {
      toast.error(`Could not delete: ${e?.message ?? e}`);
    }
  };

  return (
    <div className="w-full h-full bg-bg-surface overflow-auto p-2 flex flex-col gap-3 text-xs">
      {logPath && !!logNotebooks?.length && (
        <div>
          <div className="text-text-dim uppercase tracking-wider mb-1">In this log</div>
          <div className="flex flex-col border border-border-subtle">
            {logNotebooks.map((nb) => (
              <button
                key={nb.id}
                className="flex items-center gap-2 text-left px-2 py-1 hover:bg-bg-elevated text-text-std"
                onClick={() => openFromLog(nb.id)}
              >
                <History className="h-3 w-3 text-text-dim" />
                {nb.title}
                <span className="ml-auto text-text-dim tabular-nums">
                  {nb.runs} {nb.runs === 1 ? "cell" : "cells"} run
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-text-dim uppercase tracking-wider">Notebooks</span>
          <Button variant="ghost" size="xs" className="ml-auto" onClick={() => create("Untitled", [""])}>
            <FilePlus className="h-3 w-3" /> New
          </Button>
        </div>
        <div className="flex flex-col border border-border-subtle">
          {(notebooks ?? []).map((nb) => (
            <div key={nb.id} className="group flex items-center gap-2 px-2 py-1 hover:bg-bg-elevated">
              <button className="flex-1 text-left text-text-std" onClick={() => open(nb)}>
                {nb.title}
              </button>
              <span className="text-text-dim tabular-nums">{new Date(nb.updated_ms).toLocaleString()}</span>
              <Button
                variant="ghost"
                size="icon-xs"
                className="opacity-0 group-hover:opacity-100 hover:text-accent-red"
                onClick={() => remove(nb.id, nb.title)}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
          {notebooks && notebooks.length === 0 && <div className="px-2 py-1 text-text-dim">No notebooks yet.</div>}
        </div>
      </div>

      <div>
        <div className="text-text-dim uppercase tracking-wider mb-1">Tutorials</div>
        <div className="flex flex-col border border-border-subtle">
          {NOTEBOOK_EXAMPLES.map((ex) => (
            <button
              key={ex.id}
              className="flex items-center gap-2 text-left px-2 py-1 hover:bg-bg-elevated text-text-std"
              onClick={() => create(ex.title, ex.cells)}
            >
              <BookOpen className="h-3 w-3 text-text-dim" />
              {ex.title}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// ----------------------------------------------------------------------------

const NotebookPanel: React.FC<IDockviewPanelProps<{ notebookId?: string }>> = (props) => {
  const id = props.params?.notebookId;
  return id ? <NotebookEditor notebookId={id} panel={props} /> : <NotebookLauncher panel={props} />;
};

export default NotebookPanel;
