import { TeamColor, UiCommand } from "@/bindings";
import { ALL_PANELS, PanelId, PANEL_IDS } from "@/components/panels";

/**
 * Central command registry. Each command is a self-contained action with a
 * display title, an optional keybinding, and a `run` that takes a
 * {@link CommandContext}. The global keyboard handler dispatches keydowns to
 * these, and a future command palette can reuse the same list + context to
 * invoke commands by id.
 */

export interface CommandContext {
  selectedPlayerId: number | null;
  setSelectedPlayerId: (id: number | null) => void;
  ownPlayerIds: number[];
  isManual: (id: number) => boolean;
  setManual: (id: number, on: boolean) => void;
  sendCommand: (c: UiCommand) => void;
  primaryTeam: TeamColor;
  toggleKeyboardControl: () => void;
  toggleKeyboardMode: () => void;
  focusPanel: (id: PanelId) => void;
  toggleCommandPalette: () => void;
  /** True when manual keyboard-driving is active (suppresses plain-letter keys). */
  drivingActive: boolean;
  /** Whether the executor is currently running. */
  executorRunning: boolean;
  feedback: (label: string) => void;
}

export interface Command {
  id: string;
  title: string;
  /** Human-readable keybinding, also parsed for keydown matching. */
  keys?: string;
  /** Don't fire on plain-letter keys while keyboard-driving a robot. */
  suppressedWhileDriving?: boolean;
  enabled?: (ctx: CommandContext) => boolean;
  run: (ctx: CommandContext) => void;
}

// --- keybinding parsing ---

interface Combo {
  key: string; // lowercase ev.key, or "escape"
  shift: boolean;
}

const parseCombo = (keys: string): Combo => {
  const parts = keys.toLowerCase().split("+");
  return {
    key: parts[parts.length - 1],
    shift: parts.includes("shift"),
  };
};

const comboMatches = (ev: KeyboardEvent, combo: Combo): boolean => {
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return false;
  if (combo.shift !== ev.shiftKey) return false;
  return ev.key.toLowerCase() === combo.key;
};

/** Find the first command whose keybinding matches the event. */
export const matchKeyboard = (ev: KeyboardEvent): Command | undefined =>
  COMMANDS.find((c) => c.keys && comboMatches(ev, parseCombo(c.keys)));

/** True if focus is in a text field where shortcuts shouldn't fire. */
export const isTypingTarget = (el: EventTarget | null): boolean => {
  const node = el as HTMLElement | null;
  if (!node) return false;
  const tag = node.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    node.isContentEditable
  );
};

// --- command list ---

export const COMMANDS: Command[] = [
  {
    id: "deselect-player",
    title: "Deselect robot",
    keys: "Escape",
    enabled: (c) => c.selectedPlayerId !== null,
    run: (c) => {
      c.setSelectedPlayerId(null);
      c.feedback("Deselect robot");
    },
  },
  {
    id: "toggle-manual",
    title: "Toggle manual control",
    keys: "Shift+M",
    enabled: (c) => c.selectedPlayerId !== null,
    run: (c) => {
      const id = c.selectedPlayerId!;
      const on = !c.isManual(id);
      c.setManual(id, on);
      c.feedback(`Manual ${on ? "on" : "off"} · #${id}`);
    },
  },
  {
    id: "toggle-keyboard-control",
    title: "Toggle keyboard control",
    keys: "Shift+K",
    run: (c) => {
      c.toggleKeyboardControl();
      c.feedback("Toggle keyboard control");
    },
  },
  {
    id: "toggle-frame",
    title: "Toggle global/local frame",
    keys: "Shift+G",
    run: (c) => {
      c.toggleKeyboardMode();
      c.feedback("Toggle global/local frame");
    },
  },
  {
    id: "start-executor",
    title: "Start executor",
    keys: "Shift+S",
    enabled: (c) => !c.executorRunning,
    run: (c) => {
      c.sendCommand({ type: "Start" });
      c.feedback("Start executor");
    },
  },
  {
    id: "kill-executor",
    title: "Kill executor",
    keys: "Shift+X",
    enabled: (c) => c.executorRunning,
    run: (c) => {
      c.sendCommand({ type: "Stop" });
      c.setSelectedPlayerId(null);
      c.feedback("Kill executor");
    },
  },
  {
    id: "focus-game-controller",
    title: "Focus Game Controller",
    keys: "C",
    suppressedWhileDriving: true,
    run: (c) => {
      c.focusPanel(PANEL_IDS.GAME_CONTROLLER);
      c.feedback("Focus Game Controller");
    },
  },
  {
    id: "focus-scenario",
    title: "Focus Scenario",
    keys: "S",
    suppressedWhileDriving: true,
    run: (c) => {
      c.focusPanel(PANEL_IDS.SCENARIO);
      c.feedback("Focus Scenario");
    },
  },
  {
    id: "focus-debug",
    title: "Focus Debug",
    keys: "D",
    suppressedWhileDriving: true,
    run: (c) => {
      c.focusPanel(PANEL_IDS.DEBUG_LAYERS);
      c.feedback("Focus Debug");
    },
  },
];

/** Extra display-only entries for the cheat sheet (handled outside COMMANDS). */
export const EXTRA_SHORTCUTS: { keys: string; title: string }[] = [
  { keys: "⌘K", title: "Command palette" },
  { keys: "0–9", title: "Select robot by id" },
  { keys: "H", title: "Set target at cursor (over field)" },
];

// --- command palette ---

export interface PaletteEntry {
  id: string;
  title: string;
  section: string;
  keys?: string;
  run: () => void;
}

/**
 * Build the flat list of runnable entries for the command palette from the
 * shared registry plus dynamic items (robots, panels). The palette and the
 * keyboard handler thus stay in sync — both driven by {@link CommandContext}.
 */
export const buildPaletteEntries = (ctx: CommandContext): PaletteEntry[] => {
  const entries: PaletteEntry[] = [];

  // Static commands (panel-focus ones are covered by the Panels section).
  for (const cmd of COMMANDS) {
    if (cmd.id.startsWith("focus-")) continue;
    if (cmd.enabled && !cmd.enabled(ctx)) continue;
    entries.push({
      id: cmd.id,
      title: cmd.title,
      section: "Commands",
      keys: cmd.keys,
      run: () => cmd.run(ctx),
    });
  }

  // Robots.
  for (const id of [...ctx.ownPlayerIds].sort((a, b) => a - b)) {
    entries.push({
      id: `select-robot-${id}`,
      title: `Select robot #${id}`,
      section: "Robots",
      run: () => {
        ctx.setSelectedPlayerId(id);
        ctx.feedback(`Select #${id}`);
      },
    });
  }

  // Panels.
  for (const panel of ALL_PANELS) {
    entries.push({
      id: `panel-${panel.id}`,
      title: `Focus ${panel.title}`,
      section: "Panels",
      run: () => {
        ctx.focusPanel(panel.id);
        ctx.feedback(`Focus ${panel.title}`);
      },
    });
  }

  return entries;
};
