import { useEffect, useRef } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { DockviewApi } from "dockview";
import { toast } from "sonner";
import {
  selectedPlayerIdAtom,
  keyboardControlAtom,
  keyboardModeAtom,
  lastShortcutAtom,
  commandPaletteOpenAtom,
  currentFrameIdAtom,
  isReplayingAtom,
  useWorldState,
  useExecutorInfo,
  useSendCommand,
  usePrimaryTeam,
  useStatus,
  isPlayerManuallyControlled,
} from "@/api";
import { TeamColor } from "@/bindings";
import { PanelId, PANEL_IDS, PANEL_TITLES } from "@/components/panels";
import { openOrFocusPanel } from "@/components/toolbar/AddPanelMenu";
import { openMarkerToast } from "@/components/MarkerToast";
import { CommandContext, isTypingTarget, matchKeyboard } from "./commands";

/** Max gap (ms) between the two spaces of the double-space marker shortcut. */
const DOUBLE_SPACE_MS = 400;

/**
 * Builds the live {@link CommandContext} from app state. Recomputed each render
 * so callers (keyboard handler, future command palette) always act on current
 * values.
 */
export function useCommandContext(
  getDockviewApi: () => DockviewApi | null
): CommandContext {
  const selectedPlayerId = useAtomValue(selectedPlayerIdAtom);
  const setSelectedPlayerId = useSetAtom(selectedPlayerIdAtom);
  const setKeyboardControl = useSetAtom(keyboardControlAtom);
  const setKeyboardMode = useSetAtom(keyboardModeAtom);
  const keyboardControl = useAtomValue(keyboardControlAtom);
  const setLastShortcut = useSetAtom(lastShortcutAtom);
  const setCommandPaletteOpen = useSetAtom(commandPaletteOpenAtom);
  const world = useWorldState();
  const executorInfo = useExecutorInfo();
  const sendCommand = useSendCommand();
  const [primaryTeam] = usePrimaryTeam();
  const { data: status } = useStatus();
  const executorRunning = status?.executor?.type === "RunningExecutor";

  const ownPlayers =
    world.status === "connected"
      ? primaryTeam === TeamColor.Blue
        ? world.data.blue_team
        : world.data.yellow_team
      : [];
  const manualList = executorInfo?.manual_controlled_players ?? [];
  const isManual = (id: number) => isPlayerManuallyControlled(id, manualList);

  const drivingActive =
    selectedPlayerId !== null && keyboardControl && isManual(selectedPlayerId);

  return {
    selectedPlayerId,
    setSelectedPlayerId,
    ownPlayerIds: ownPlayers.map((p) => p.id),
    isManual,
    setManual: (id, on) =>
      sendCommand({
        type: "SetManualOverride",
        data: { team_color: primaryTeam, player_id: id, manual_override: on },
      }),
    sendCommand,
    primaryTeam,
    toggleKeyboardControl: () => setKeyboardControl((v) => !v),
    toggleKeyboardMode: () =>
      setKeyboardMode((m) => (m === "global" ? "local" : "global")),
    focusPanel: (id: PanelId) => {
      const api = getDockviewApi();
      if (api) openOrFocusPanel(api, id, PANEL_TITLES[id]);
    },
    toggleCommandPalette: () => setCommandPaletteOpen((v) => !v),
    drivingActive,
    executorRunning,
    feedback: (label: string) => setLastShortcut({ label, ts: Date.now() }),
  };
}

/**
 * Registers the global keyboard shortcut handler. Number keys select a robot by
 * id; everything else dispatches through the shared command registry.
 */
export function useGlobalShortcuts(getDockviewApi: () => DockviewApi | null) {
  const ctx = useCommandContext(getDockviewApi);
  const ctxRef = useRef(ctx);
  ctxRef.current = ctx;

  // Live current-frame + replay flag, read by the double-space marker handler.
  const currentFrameId = useAtomValue(currentFrameIdAtom);
  const isReplaying = useAtomValue(isReplayingAtom);
  const frameRef = useRef(currentFrameId);
  frameRef.current = currentFrameId;
  const replayingRef = useRef(isReplaying);
  replayingRef.current = isReplaying;
  const lastSpaceRef = useRef(0);

  useEffect(() => {
    // True for the command-palette combo (⌘K / Ctrl+K, no other modifiers).
    const isPaletteCombo = (ev: KeyboardEvent) =>
      (ev.metaKey || ev.ctrlKey) &&
      !ev.shiftKey &&
      !ev.altKey &&
      ev.key.toLowerCase() === "k";

    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.repeat) return;
      const c = ctxRef.current;

      // Command palette (⌘K / Ctrl+K) — works even while typing. Capture phase
      // + preventDefault to beat Firefox's search-bar accelerator.
      if (isPaletteCombo(ev)) {
        ev.preventDefault();
        ev.stopPropagation();
        c.toggleCommandPalette();
        return;
      }

      if (isTypingTarget(document.activeElement)) return;

      // Double-space drops a point-of-interest marker (live recording only).
      // The second space within the window opens the label toast.
      if (
        ev.code === "Space" &&
        !ev.shiftKey &&
        !ev.ctrlKey &&
        !ev.metaKey &&
        !ev.altKey
      ) {
        const now = ev.timeStamp;
        if (now - lastSpaceRef.current < DOUBLE_SPACE_MS) {
          lastSpaceRef.current = 0;
          ev.preventDefault();
          if (!replayingRef.current) {
            openMarkerToast((label) => {
              c.sendCommand({
                type: "AddMarker",
                data: { label: label ?? undefined },
              });
              const f = Math.round(frameRef.current / 50) * 50;
              toast.success(
                label ? `Marker "${label}" @ f≈${f}` : `Marker @ f≈${f}`
              );
            });
          }
          return;
        }
        lastSpaceRef.current = now;
        return;
      }

      // Number keys: quick-switch to robot by id.
      if (
        !ev.shiftKey &&
        !ev.ctrlKey &&
        !ev.metaKey &&
        !ev.altKey &&
        /^[0-9]$/.test(ev.key)
      ) {
        const id = Number(ev.key);
        if (c.ownPlayerIds.includes(id)) {
          c.setSelectedPlayerId(id);
          c.feedback(`Select #${id}`);
        }
        return;
      }

      const cmd = matchKeyboard(ev);
      if (!cmd) return;
      if (cmd.suppressedWhileDriving && c.drivingActive) return;
      if (cmd.enabled && !cmd.enabled(c)) return;
      ev.preventDefault();
      cmd.run(c);
    };

    // Firefox fires the search-bar accelerator for Ctrl/Cmd+K; swallow the
    // matching keypress/keyup too so only the palette opens.
    const swallowPalette = (ev: KeyboardEvent) => {
      if (isPaletteCombo(ev)) {
        ev.preventDefault();
        ev.stopPropagation();
      }
    };

    window.addEventListener("keydown", onKeyDown, { capture: true });
    window.addEventListener("keypress", swallowPalette, { capture: true });
    window.addEventListener("keyup", swallowPalette, { capture: true });
    return () => {
      window.removeEventListener("keydown", onKeyDown, { capture: true });
      window.removeEventListener("keypress", swallowPalette, { capture: true });
      window.removeEventListener("keyup", swallowPalette, { capture: true });
    };
  }, []);
}

// Re-export so the future command palette can pull panel ids if needed.
export { PANEL_IDS };
