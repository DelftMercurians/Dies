import React from "react";
import { Hand, Play, RectangleVertical, ShieldOff } from "lucide-react";

import {
  useExecutorSettings,
  useSendCommand,
  useStatus,
  useWorldState,
} from "@/api";
import { GcSimCommand, TeamColor, UiMode } from "@/bindings";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Game-controller quick actions, centered in the toolbar above the score
 * banner:
 *
 *  - Stop | Continue (simulation only) — drive the sim's auto-ref.
 *  - Ignore-GC per-team toggles (sim + live) — a testing override that makes a
 *    team play as if in free play regardless of the actual GC state. Continue
 *    resumes the right way — a prepared kick-off/penalty needs a normal start,
 *    anything else just force-starts open play.
 */
const GcControls: React.FC = () => {
  const { data: backendState } = useStatus();
  const world = useWorldState();
  const sendCommand = useSendCommand();
  const { settings, updateSettings } = useExecutorSettings();

  const isSim = backendState?.ui_mode === UiMode.Simulation;

  const send = (command: GcSimCommand) =>
    sendCommand({ type: "GcCommand", data: command });

  const continueCommand = (): GcSimCommand => {
    const t =
      world.status === "connected"
        ? world.data.game_state.game_state.type
        : undefined;
    if (t === "PrepareKickoff" || t === "PreparePenalty") {
      return { type: "NormalStart" };
    }
    return { type: "ForceStart" };
  };

  const blueIgnore = settings?.blue_team_settings.ignore_gc ?? false;
  const yellowIgnore = settings?.yellow_team_settings.ignore_gc ?? false;

  const toggleIgnore = (team: "blue" | "yellow", value: boolean) => {
    if (!settings) return;
    updateSettings(
      team === "blue"
        ? {
            ...settings,
            blue_team_settings: {
              ...settings.blue_team_settings,
              ignore_gc: value,
            },
          }
        : {
            ...settings,
            yellow_team_settings: {
              ...settings.yellow_team_settings,
              ignore_gc: value,
            },
          },
    );
  };

  return (
    <div className="inline-flex items-center gap-1.5">
      {isSim && (
        <div className="inline-flex items-center border border-border-muted h-7 bg-bg-surface">
          <SimpleTooltip title="Stop the game (GC)" className="h-full">
            <button
              onClick={() => send({ type: "Stop" })}
              className={cn(
                "h-full w-7 flex items-center justify-center transition-colors",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
                "text-text-muted hover:text-accent-red hover:bg-accent-red/10",
              )}
            >
              <Hand className="w-3.5 h-3.5" />
            </button>
          </SimpleTooltip>

          <div className="w-px h-full bg-border-subtle" />

          <SimpleTooltip title="Continue / resume play (GC)" className="h-full">
            <button
              onClick={() => send(continueCommand())}
              className={cn(
                "h-full w-7 flex items-center justify-center transition-colors",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
                "text-text-muted hover:text-accent-green hover:bg-accent-green/10",
              )}
            >
              <Play className="w-3.5 h-3.5 fill-current" />
            </button>
          </SimpleTooltip>
        </div>
      )}

      {/* Yellow-card injection (sim only): lowers the team's max_allowed_bots
          for 120s, forcing a robot to be sidelined — for testing removal. */}
      {isSim && (
        <div className="inline-flex items-center border border-border-muted h-7 bg-bg-surface">
          <SimpleTooltip
            title="Show blue a yellow card (sim) — sidelines a robot for 120s"
            className="h-full"
          >
            <button
              onClick={() =>
                send({ type: "YellowCard", data: { team_color: TeamColor.Blue } })
              }
              className={cn(
                "h-full px-2 flex items-center gap-1 text-[10px] font-semibold tracking-wide transition-colors",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
                "text-text-muted hover:text-team-blue hover:bg-team-blue/10",
              )}
            >
              <RectangleVertical className="w-3 h-3" />B
            </button>
          </SimpleTooltip>

          <div className="w-px h-full bg-border-subtle" />

          <SimpleTooltip
            title="Show yellow a yellow card (sim) — sidelines a robot for 120s"
            className="h-full"
          >
            <button
              onClick={() =>
                send({
                  type: "YellowCard",
                  data: { team_color: TeamColor.Yellow },
                })
              }
              className={cn(
                "h-full px-2 flex items-center gap-1 text-[10px] font-semibold tracking-wide transition-colors",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
                "text-text-muted hover:text-team-yellow hover:bg-team-yellow/10",
              )}
            >
              <RectangleVertical className="w-3 h-3" />Y
            </button>
          </SimpleTooltip>
        </div>
      )}

      {/* Ignore-GC per-team override (sim + live): when on, the team plays as
          if in free play regardless of GC state. */}
      <div className="inline-flex items-center border border-border-muted h-7 bg-bg-surface">
        <SimpleTooltip
          title={
            blueIgnore
              ? "Blue is IGNORING the GC (free play) — click to obey"
              : "Blue obeys the GC — click to ignore (free play, for testing)"
          }
          className="h-full"
        >
          <button
            onClick={() => toggleIgnore("blue", !blueIgnore)}
            disabled={!settings}
            className={cn(
              "h-full px-2 flex items-center gap-1 text-[10px] font-semibold tracking-wide transition-colors",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
              "disabled:opacity-40 disabled:cursor-not-allowed",
              blueIgnore
                ? "bg-team-blue/20 text-team-blue"
                : "text-text-muted hover:text-team-blue hover:bg-team-blue/10",
            )}
          >
            <ShieldOff className="w-3 h-3" />B
          </button>
        </SimpleTooltip>

        <div className="w-px h-full bg-border-subtle" />

        <SimpleTooltip
          title={
            yellowIgnore
              ? "Yellow is IGNORING the GC (free play) — click to obey"
              : "Yellow obeys the GC — click to ignore (free play, for testing)"
          }
          className="h-full"
        >
          <button
            onClick={() => toggleIgnore("yellow", !yellowIgnore)}
            disabled={!settings}
            className={cn(
              "h-full px-2 flex items-center gap-1 text-[10px] font-semibold tracking-wide transition-colors",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
              "disabled:opacity-40 disabled:cursor-not-allowed",
              yellowIgnore
                ? "bg-team-yellow/20 text-team-yellow"
                : "text-text-muted hover:text-team-yellow hover:bg-team-yellow/10",
            )}
          >
            <ShieldOff className="w-3 h-3" />Y
          </button>
        </SimpleTooltip>
      </div>
    </div>
  );
};

export default GcControls;
