import React from "react";
import { useStatus, useSetMode } from "@/api";
import { UiMode } from "@/bindings";
import { cn } from "@/lib/utils";

/**
 * Compact mode toggle: SIM | LIV
 *
 * Specs:
 * - Two-segment toggle (~64px total)
 * - Active segment: --accent-green background
 * - Disabled state for unavailable live mode
 */

const ModeToggle: React.FC = () => {
  const { data: backendState } = useStatus();
  const { mutate: setMode } = useSetMode();

  const currentMode = backendState?.ui_mode ?? UiMode.Simulation;
  const isLiveAvailable = backendState?.is_live_available ?? false;

  const handleModeChange = (mode: UiMode) => {
    if (mode === UiMode.Live && !isLiveAvailable) return;
    setMode(mode);
  };

  return (
    <div className="inline-flex items-center border border-border-muted h-5">
      <button
        onClick={() => handleModeChange(UiMode.Simulation)}
        className={cn(
          "h-full px-2 text-[9px] font-semibold uppercase tracking-wider transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
          currentMode === UiMode.Simulation
            ? "bg-accent-green text-bg-base"
            : "bg-transparent text-text-muted hover:text-text-std hover:bg-bg-overlay"
        )}
      >
        SIM
      </button>
      <div className="w-px h-full bg-border-subtle" />
      <button
        onClick={() => handleModeChange(UiMode.Live)}
        disabled={!isLiveAvailable}
        className={cn(
          "h-full px-2 text-[9px] font-semibold uppercase tracking-wider transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
          "disabled:opacity-40 disabled:cursor-not-allowed",
          currentMode === UiMode.Live
            ? "bg-accent-green text-bg-base"
            : "bg-transparent text-text-muted hover:text-text-std hover:bg-bg-overlay"
        )}
      >
        LIV
      </button>
    </div>
  );
};

export default ModeToggle;

