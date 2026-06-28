import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import PlayerSidebar from "@/views/PlayerSidebar";
import TeamOverview from "@/views/TeamOverview";
import { selectedPlayerIdAtom } from "@/api";

/**
 * Inspector Panel — the right sidebar. With no player selected it shows the
 * team overview (one dense row per own-team player, including any active
 * pass-coordinator diagnostics); selecting a player shows the detailed sidebar
 * (hardware, debug, plot, manual control).
 */
const PlayerInspectorPanel: React.FC<IDockviewPanelProps> = () => {
  const [selectedPlayerId, setSelectedPlayerId] = useAtom(selectedPlayerIdAtom);

  return (
    <div className="w-full h-full bg-bg-surface overflow-hidden flex flex-col">
      <div className="flex-1 overflow-hidden">
        {selectedPlayerId === null ? (
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
