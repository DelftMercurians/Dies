import React from "react";
import { Hand, Play } from "lucide-react";

import { useSendCommand, useStatus, useWorldState } from "@/api";
import { GcSimCommand, UiMode } from "@/bindings";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Game-controller quick actions (simulation only): Stop | Continue.
 *
 * Continue resumes the right way — a prepared kick-off/penalty needs a normal
 * start, anything else just force-starts open play.
 */
const GcControls: React.FC = () => {
  const { data: backendState } = useStatus();
  const world = useWorldState();
  const sendCommand = useSendCommand();

  if (backendState?.ui_mode !== UiMode.Simulation) return null;

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

  return (
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
  );
};

export default GcControls;
