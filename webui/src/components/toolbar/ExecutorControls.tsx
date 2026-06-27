import React from "react";
import { Play, Pause, Square } from "lucide-react";
import { useStatus, useSendCommand, useExecutorInfo } from "@/api";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Executor control buttons: Play | Pause | Stop
 *
 * Specs:
 * - Icon buttons, 20x20 each in a grouped container
 * - Color states: green running, amber paused, red stop hover
 */

interface ExecutorControlsProps {
  onStart?: () => void;
  onStop?: () => void;
}

const ExecutorControls: React.FC<ExecutorControlsProps> = ({
  onStart,
  onStop,
}) => {
  const { data: backendState } = useStatus();
  const sendCommand = useSendCommand();
  const executorInfo = useExecutorInfo();

  const executorStatus = backendState?.executor;
  const isRunning = executorStatus?.type === "RunningExecutor";
  const isSim = backendState?.ui_mode === "Simulation";
  // Pause is only supported on the simulator (a live match can't be frozen).
  const canPause = isRunning && isSim;
  const isPaused = canPause && (executorInfo?.paused ?? false);

  const handlePlay = () => {
    if (!isRunning) {
      onStart?.();
      sendCommand({ type: "Start" });
    }
  };

  const handlePause = () => {
    if (!canPause) return;
    sendCommand({ type: "SetPause", data: !isPaused });
  };

  const handleStop = () => {
    if (isRunning) {
      onStop?.();
      sendCommand({ type: "Stop" });
    }
  };

  return (
    <div className="inline-flex items-center border border-border-muted h-7">
      {/* Play */}
      <SimpleTooltip title="Start executor" className="h-full">
        <button
          onClick={handlePlay}
          className={cn(
            "h-full w-7 flex items-center justify-center transition-colors",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
            isRunning
              ? "bg-accent-green/20 text-accent-green"
              : "text-text-muted hover:text-accent-green hover:bg-accent-green/10",
          )}
        >
          <Play className="w-3.5 h-3.5 fill-current" />
        </button>
      </SimpleTooltip>

      <div className="w-px h-full bg-border-subtle" />

      {/* Pause */}
      <SimpleTooltip
        title={isPaused ? "Resume executor" : "Pause executor"}
        className="h-full"
      >
        <button
          onClick={handlePause}
          disabled={!canPause}
          className={cn(
            "h-full w-7 flex items-center justify-center transition-colors",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
            "disabled:opacity-40 disabled:cursor-not-allowed",
            isPaused
              ? "bg-accent-amber/20 text-accent-amber"
              : "text-text-muted hover:text-accent-amber hover:bg-accent-amber/10",
          )}
        >
          <Pause className="w-3.5 h-3.5 fill-current" />
        </button>
      </SimpleTooltip>

      <div className="w-px h-full bg-border-subtle" />

      {/* Stop */}
      <SimpleTooltip title="Stop executor" className="h-full">
        <button
          onClick={handleStop}
          disabled={!isRunning}
          className={cn(
            "h-full w-7 flex items-center justify-center transition-colors",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
            "disabled:opacity-40 disabled:cursor-not-allowed",
            "text-text-muted hover:text-accent-red hover:bg-accent-red/10",
          )}
        >
          <Square className="w-3.5 h-3.5" />
        </button>
      </SimpleTooltip>
    </div>
  );
};

export default ExecutorControls;
