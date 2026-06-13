import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import PlayerSidebar from "@/views/PlayerSidebar";
import TeamOverview from "@/views/TeamOverview";
import { selectedPlayerIdAtom } from "@/api";

/**
 * Inspector Panel.
 *
 * - No player selected: shows a superdense, glanceable team overview.
 * - Player selected: shows the detailed player sidebar (hardware, debug, plot,
 *   manual control), with a quick switcher to jump between players.
 */
const PlayerInspectorPanel: React.FC<IDockviewPanelProps> = () => {
  const [selectedPlayerId, setSelectedPlayerId] = useAtom(selectedPlayerIdAtom);

  return (
    <div className="w-full h-full bg-bg-surface overflow-hidden flex flex-col">
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
  );
};

export default PlayerInspectorPanel;
