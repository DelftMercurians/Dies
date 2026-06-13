import { useEffect, useRef } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { DockviewApi } from "dockview";
import {
  selectedPlayerIdAtom,
  keyboardControlAtom,
  keyboardModeAtom,
  lastShortcutAtom,
  useWorldState,
  useExecutorInfo,
  useSendCommand,
  usePrimaryTeam,
  isPlayerManuallyControlled,
} from "@/api";
import { TeamColor } from "@/bindings";
import { PanelId, PANEL_IDS, PANEL_TITLES } from "@/components/panels";
import { openOrFocusPanel } from "@/components/toolbar/AddPanelMenu";
import { CommandContext, isTypingTarget, matchKeyboard } from "./commands";

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
  const world = useWorldState();
  const executorInfo = useExecutorInfo();
  const sendCommand = useSendCommand();
  const [primaryTeam] = usePrimaryTeam();

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
    drivingActive,
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

  useEffect(() => {
    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.repeat) return;
      if (isTypingTarget(document.activeElement)) return;
      const c = ctxRef.current;

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

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}

// Re-export so the future command palette can pull panel ids if needed.
export { PANEL_IDS };
