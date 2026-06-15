import React, { useState } from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import PlayerSidebar from "@/views/PlayerSidebar";
import TeamOverview from "@/views/TeamOverview";
import PassInspector from "@/views/PassInspector";
import { selectedPlayerIdAtom } from "@/api";
import { cn } from "@/lib/utils";

type Mode = "inspect" | "pass";

/**
 * Inspector Panel — the right sidebar, with a mode toggle:
 *
 * - "Inspect": team overview (no player selected) or the detailed player
 *   sidebar (hardware, debug, plot, manual control).
 * - "Pass": dedicated view of every active pass coordinator's internal state.
 */
const PlayerInspectorPanel: React.FC<IDockviewPanelProps> = () => {
  const [selectedPlayerId, setSelectedPlayerId] = useAtom(selectedPlayerIdAtom);
  const [mode, setMode] = useState<Mode>("inspect");

  return (
    <div className="w-full h-full bg-bg-surface overflow-hidden flex flex-col">
      <div className="flex items-center gap-1 p-1 border-b border-border">
        {(["inspect", "pass"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={cn(
              "px-2 py-0.5 rounded text-xs capitalize",
              mode === m
                ? "bg-bg-base text-text font-medium"
                : "text-text-muted hover:text-text"
            )}
          >
            {m}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-hidden">
        {mode === "pass" ? (
          <PassInspector />
        ) : selectedPlayerId === null ? (
          <TeamOverview
            onSelectPlayer={setSelectedPlayerId}
            selectedPlayerId={selectedPlayerId}
            className="h-full"
          />
        ) : (
          <PlayerSidebar
            selectedPlayerId={selectedPlayerId}
            onClose={() => setSelectedPlayerId(null)}
            onSelectPlayer={setSelectedPlayerId}
          />
        )}
      </div>
    </div>
  );
};

export default PlayerInspectorPanel;
